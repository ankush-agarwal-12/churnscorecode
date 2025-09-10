import asyncio, json, sounddevice as sd, numpy as np
from deepgram import DeepgramClient, DeepgramClientOptions, LiveTranscriptionEvents
from deepgram.clients.live.v1 import LiveOptions
import queue
import threading
from datetime import datetime
from simple_churn_scorer import SimpleChurnScorer
import warnings
warnings.filterwarnings("ignore")

# Your Deepgram API Key
DEEPGRAM_API_KEY = "ae4c98bf98d958f94a902320f592351582a35c30"

# Audio config
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 1024  # ~64ms

# Audio queue for passing data between threads
audio_queue = queue.Queue()

class SimpleLiveChurnDetector:
    def __init__(self):
        self.churn_scorer = SimpleChurnScorer()
        self.current_speaker = None
        self.last_speaker = None
        
        # Message accumulation
        self.current_agent_message = ""  # Accumulate agent message parts
        self.current_customer_message = ""  # Accumulate customer message parts
        self.complete_agent_context = ""  # Store complete agent message for context
        
        self.last_update_time = datetime.now()
        self.speaker_turn_counter = 0  # To alternate speakers
        
        # Simple speaker patterns for basic detection
        self.agent_patterns = [
            "thank you for calling", "customer service", "this is", "speaking",
            "i can help", "let me check", "i understand", "i apologize",
            "i see here", "i'll", "we can offer", "would you like"
        ]
        
        self.customer_patterns = [
            "my bill", "i want to", "why is", "i'm calling", "i have a problem",
            "i just", "i need", "can you", "what about", "how much"
        ]

    def determine_speaker_simple(self, text):
        """Simple speaker detection based on patterns"""
        text_lower = text.lower()
        
        agent_score = sum(1 for pattern in self.agent_patterns if pattern in text_lower)
        customer_score = sum(1 for pattern in self.customer_patterns if pattern in text_lower)
        
        if agent_score > customer_score:
            return "Agent"
        elif customer_score > 0:
            return "Customer"
        else:
            # Simple alternating logic as fallback
            self.speaker_turn_counter += 1
            return "Agent" if self.speaker_turn_counter % 2 == 1 else "Customer"

    def process_transcript(self, transcript_data, speaker_id=None):
        """Process incoming transcript and accumulate messages until speaker changes"""
        try:
            if not transcript_data or not transcript_data.strip():
                return
                
            # Determine speaker
            if speaker_id is not None:
                speaker = f"Speaker {speaker_id}"
                # Map to Customer/Agent based on ID (0 usually = first speaker)
                if speaker_id == 0:
                    speaker = "Agent"  # Assuming agent speaks first
                else:
                    speaker = "Customer"
            else:
                speaker = self.determine_speaker_simple(transcript_data)
            
            # Check if speaker has changed
            speaker_changed = self.current_speaker != speaker
            
            # If speaker changed, process the complete previous message
            if speaker_changed and self.current_speaker is not None:
                self.process_complete_message()
            
            # Update current speaker
            self.current_speaker = speaker
            
            # Accumulate message based on current speaker (no intermediate prints)
            if speaker.lower() == "agent":
                # Accumulate agent message parts
                if speaker_changed:
                    self.current_agent_message = transcript_data
                else:
                    self.current_agent_message += " " + transcript_data
                
            elif speaker.lower() == "customer":
                # Accumulate customer message parts
                if speaker_changed:
                    self.current_customer_message = transcript_data
                else:
                    self.current_customer_message += " " + transcript_data
            
        except Exception as e:
            print(f"Error processing transcript: {e}")
    
    def process_complete_message(self):
        """Process complete message when speaker changes"""
        try:
            if self.current_speaker.lower() == "agent" and self.current_agent_message.strip():
                # Store complete agent message as context for next customer message
                self.complete_agent_context = self.current_agent_message.strip()
                print(f"\nAgent: {self.complete_agent_context}")
                
                # Clear current agent message
                self.current_agent_message = ""
                
            elif self.current_speaker.lower() == "customer" and self.current_customer_message.strip():
                # Process complete customer message with agent context
                customer_text = self.current_customer_message.strip()
                print(f"\nCustomer: {customer_text}")
                
                # Process with churn scorer using agent context
                # Agent context is passed but only customer message is analyzed for sentiment and patterns
                event = self.churn_scorer.process_customer_message(customer_text, self.complete_agent_context)
                
                # Display update with churn calculation in one line
                self.display_customer_update(customer_text, event)
                
                # Clear current customer message
                self.current_customer_message = ""
                
        except Exception as e:
            print(f"Error processing complete message: {e}")
    
    def display_customer_update(self, customer_text, event):
        """Display churn scoring update for completed customer messages"""
        current_score = self.churn_scorer.get_current_score()
        
        # Show churn calculation in one line
        if event.risk_delta != 0:
            change_indicator = "+" if event.risk_delta > 0 else ""
            print(f"Churn Score: {current_score}/100 (Risk Change: {change_indicator}{event.risk_delta:.1f})")
        else:
            print(f"Churn Score: {current_score}/100 (No Change)")

async def main():
    try:
        # Initialize live churn detector
        churn_detector = SimpleLiveChurnDetector()
        
        # Create Deepgram client
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram = DeepgramClient(DEEPGRAM_API_KEY, config)
        
        # Create a websocket connection to Deepgram
        dg_connection = deepgram.listen.asyncwebsocket.v("1")
        
        # Configure live transcription options
        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            interim_results=True,
            diarize=True,
            encoding="linear16",
            channels=CHANNELS,
            sample_rate=SAMPLE_RATE,
        )

        # Event handlers
        async def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            if len(sentence) == 0:
                return
            
            # Only process final results for churn scoring
            if result.is_final:
                speaker_id = None
                
                # Try to extract speaker information from diarization
                if hasattr(result.channel.alternatives[0], 'words') and result.channel.alternatives[0].words:
                    first_word = result.channel.alternatives[0].words[0]
                    if hasattr(first_word, 'speaker'):
                        speaker_id = first_word.speaker
                
                # Process with churn detector
                churn_detector.process_transcript(sentence, speaker_id)
            # Hide interim results - no print for interim
            # else:
            #     # Show interim results without processing
            #     print(f"[Interim] {sentence}")

        async def on_error(self, error, **kwargs):
            print(f"Error: {error}")

        async def on_close(self, close, **kwargs):
            print("Connection closed")

        # Register event handlers
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)
        dg_connection.on(LiveTranscriptionEvents.Close, on_close)

        # Start the connection
        if await dg_connection.start(options) is False:
            print("Failed to connect to Deepgram")
            return

        print("LIVE CHURN DETECTION STARTED")
        print("Listening for customer service conversations...")
        print("Real-time churn scoring active")
        print("\nSpeak or play audio. Press Ctrl+C to stop.\n")

        # Audio capture callback - runs in a separate thread
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            
            # Convert audio to bytes and put in queue
            audio_data = indata.astype(np.int16).tobytes()
            try:
                audio_queue.put_nowait(audio_data)
            except queue.Full:
                print("Audio queue full, dropping frame")

        # Function to process audio queue
        async def process_audio_queue():
            while True:
                try:
                    # Get audio data from queue (non-blocking)
                    audio_data = audio_queue.get_nowait()
                    await dg_connection.send(audio_data)
                except queue.Empty:
                    # No audio data available, wait a bit
                    await asyncio.sleep(0.01)
                except Exception as e:
                    print(f"Error sending audio: {e}")

        # Start audio capture in a separate thread
        audio_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=BLOCK_SIZE,
            callback=audio_callback
        )
        
        audio_stream.start()
        
        # Start the audio processing task
        audio_task = asyncio.create_task(process_audio_queue())
        
        try:
            # Keep the connection alive
            while True:
                await asyncio.sleep(0.1)
                
                # No periodic updates (removed as requested)
                    
        except KeyboardInterrupt:
            print("\nStopping live churn detection...")
        finally:
            # Clean up
            audio_stream.stop()
            audio_stream.close()
            audio_task.cancel()
            await dg_connection.finish()
            
            # Process any remaining complete message before ending
            if churn_detector.current_speaker:
                churn_detector.process_complete_message()
            
            # Final report
            print(f"\n{'='*50}")
            print("FINAL CONVERSATION SUMMARY")
            print("="*50)
            print(f"Final Churn Score: {churn_detector.churn_scorer.get_current_score()}/100")
            print(f"Total Customer Messages Processed: {len(churn_detector.churn_scorer.risk_events)}")
            print(f"Conversation Duration: {(datetime.now() - churn_detector.last_update_time).total_seconds():.0f} seconds")

    except Exception as e:
        print(f"Could not start live churn detection: {e}")

if __name__ == "__main__":
    # Test mode - run the example conversation
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Running test mode with example conversation...")
        from simple_churn_scorer import simulate_conversation_example
        simulate_conversation_example()
    else:
        print("Starting live mode...")
        asyncio.run(main()) 