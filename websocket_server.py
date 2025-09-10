import asyncio
import json
import websockets
import threading
import queue
from datetime import datetime
from simple_live_churn import SimpleLiveChurnDetector
import sounddevice as sd
import numpy as np
from deepgram import DeepgramClient, DeepgramClientOptions, LiveTranscriptionEvents
from deepgram.clients.live.v1 import LiveOptions
import warnings
warnings.filterwarnings("ignore")

# Your Deepgram API Key
DEEPGRAM_API_KEY = "ae4c98bf98d958f94a902320f592351582a35c30"

# Audio config
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 1024

class WebSocketChurnServer:
    def __init__(self):
        self.churn_detector = None
        self.deepgram_connection = None
        self.audio_stream = None
        self.audio_queue = queue.Queue()
        self.websocket_clients = set()
        self.is_recording = False
        self.audio_task = None
        self.transcript_messages = []  # Track transcript messages for note generation
        
    async def register_client(self, websocket):
        """Register a new WebSocket client"""
        self.websocket_clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.websocket_clients)}")
        
    async def unregister_client(self, websocket):
        """Unregister a WebSocket client"""
        self.websocket_clients.discard(websocket)
        print(f"Client disconnected. Total clients: {len(self.websocket_clients)}")
        
    async def broadcast_to_clients(self, message):
        """Broadcast message to all connected clients"""
        if self.websocket_clients:
            print(f"Broadcasting to {len(self.websocket_clients)} client(s): {message['type']}")
            disconnected_clients = set()
            for client in self.websocket_clients.copy():
                try:
                    await client.send(json.dumps(message))
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
                except Exception as e:
                    print(f"Error sending message to client: {e}")
            
            # Remove disconnected clients
            for client in disconnected_clients:
                await self.unregister_client(client)
        else:
            print(f"No clients connected to broadcast message: {message['type']}")
    
    async def initialize_models(self):
        """Initialize all ML models (called once at startup)"""
        print("Initializing churn detection models...")
        self.churn_detector = SimpleLiveChurnDetector()
        
        # Send initialization complete message
        await self.broadcast_to_clients({
            'type': 'models_initialized',
            'message': 'All models loaded successfully'
        })
        print("Models initialized successfully!")
        
    async def start_recording(self, clear_data=False):
        """Start audio recording and transcription"""
        if self.is_recording:
            return
            
        try:
            print("Starting recording...")
            
            # Clear conversation data if requested
            if clear_data and self.churn_detector:
                print("Clearing conversation data...")
                self.churn_detector.churn_scorer.reset_conversation()
                await self.broadcast_to_clients({
                    'type': 'conversation_cleared',
                    'message': 'Conversation data cleared'
                })
            
            self.is_recording = True
            
            # Create Deepgram client
            config = DeepgramClientOptions(options={"keepalive": "true"})
            deepgram = DeepgramClient(DEEPGRAM_API_KEY, config)
            
            # Create websocket connection to Deepgram
            self.deepgram_connection = deepgram.listen.asyncwebsocket.v("1")
            
            # Configure options
            options = LiveOptions(
                model="nova-2",
                punctuate=True,
                interim_results=True,
                diarize=True,
                encoding="linear16",
                channels=CHANNELS,
                sample_rate=SAMPLE_RATE,
            )
            
            # Event handlers - capture self reference
            server_instance = self
            
            async def on_message(self, result, **kwargs):
                sentence = result.channel.alternatives[0].transcript
                if len(sentence) == 0:
                    return
                
                # print(f"[Transcript] {sentence} (final: {result.is_final})")
                
                if result.is_final:
                    speaker_id = None
                    
                    # Extract speaker information
                    if hasattr(result.channel.alternatives[0], 'words') and result.channel.alternatives[0].words:
                        first_word = result.channel.alternatives[0].words[0]
                        if hasattr(first_word, 'speaker'):
                            speaker_id = first_word.speaker
                    
                    # Process with churn detector
                    # print(f"About to call process_transcript with: {sentence}")
                    await server_instance.process_transcript(sentence, speaker_id)
            
            async def on_error(self, error, **kwargs):
                print(f"Deepgram error: {error}")
                await server_instance.broadcast_to_clients({
                    'type': 'error',
                    'message': f"Transcription error: {error}"
                })
            
            async def on_close(self, close, **kwargs):
                print("Deepgram connection closed")
            
            # Register event handlers
            self.deepgram_connection.on(LiveTranscriptionEvents.Transcript, on_message)
            self.deepgram_connection.on(LiveTranscriptionEvents.Error, on_error)
            self.deepgram_connection.on(LiveTranscriptionEvents.Close, on_close)
            
            # Start connection
            if await self.deepgram_connection.start(options) is False:
                raise Exception("Failed to connect to Deepgram")
            
            # Start audio capture
            await self.start_audio_capture()
            
            # Send recording started message
            await self.broadcast_to_clients({
                'type': 'recording_started',
                'message': 'Recording started successfully'
            })
            
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.is_recording = False
            await self.broadcast_to_clients({
                'type': 'error',
                'message': f"Failed to start recording: {e}"
            })
    
    async def stop_recording(self):
        """Stop recording and reset conversation history"""
        if not self.is_recording:
            return
            
        try:
            print("Stopping recording...")
            
            # Process any final queued message before stopping
            if self.churn_detector and self.churn_detector.current_speaker:
                print("Processing final queued message...")
                await self.process_complete_message()
            
            self.is_recording = False
            
            # Stop audio stream
            if self.audio_stream:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None
            
            # Cancel audio task
            if self.audio_task:
                self.audio_task.cancel()
                self.audio_task = None
            
            # Close Deepgram connection
            if self.deepgram_connection:
                await self.deepgram_connection.finish()
                self.deepgram_connection = None
            
            # Don't reset conversation history on stop - only clear current message buffers
            if self.churn_detector:
                self.churn_detector.current_speaker = None
                self.churn_detector.current_agent_message = ""
                self.churn_detector.current_customer_message = ""
                self.churn_detector.complete_agent_context = ""
            
            # Clear audio queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Send recording stopped message
            await self.broadcast_to_clients({
                'type': 'recording_stopped',
                'message': 'Recording stopped and conversation reset'
            })
            
        except Exception as e:
            print(f"Error stopping recording: {e}")
            await self.broadcast_to_clients({
                'type': 'error',
                'message': f"Failed to stop recording: {e}"
            })
    
    async def start_audio_capture(self):
        """Start audio capture in a separate thread"""
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            
            if self.is_recording:
                audio_data = indata.astype(np.int16).tobytes()
                try:
                    self.audio_queue.put_nowait(audio_data)
                    # Add periodic logging to see if audio is being captured
                    if hasattr(self, '_audio_frame_count'):
                        self._audio_frame_count += 1
                        if self._audio_frame_count % 100 == 0:  # Log every 100 frames (~6.4 seconds)
                            print(f"Audio frames captured: {self._audio_frame_count}")
                    else:
                        self._audio_frame_count = 1
                        print("Audio capture started - first frame captured")
                except queue.Full:
                    print("Audio queue full, dropping frame")
        
        # Start audio stream
        self.audio_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=BLOCK_SIZE,
            callback=audio_callback
        )
        self.audio_stream.start()
        print(f"Audio stream started: {SAMPLE_RATE}Hz, {CHANNELS} channel(s)")
        
        # Start audio processing task
        self.audio_task = asyncio.create_task(self.process_audio_queue())
    
    async def process_audio_queue(self):
        """Process audio queue and send to Deepgram"""
        sent_frames = 0
        while self.is_recording:
            try:
                audio_data = self.audio_queue.get_nowait()
                if self.deepgram_connection:
                    await self.deepgram_connection.send(audio_data)
                    sent_frames += 1
                    if sent_frames % 100 == 0:  # Log every 100 frames
                        print(f"Audio frames sent to Deepgram: {sent_frames}")
            except queue.Empty:
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"Error processing audio: {e}")
    
    async def process_transcript(self, transcript_data, speaker_id=None):
        """Process transcript and send updates to frontend"""
        try:
            if not self.churn_detector or not transcript_data.strip():
                return
            
            # Determine speaker
            if speaker_id is not None:
                speaker = "Agent" if speaker_id == 0 else "Customer"
            else:
                speaker = self.churn_detector.determine_speaker_simple(transcript_data)
            
            print(f"Processing transcript - Speaker: {speaker}, Text: {transcript_data}")
            
            # Send individual transcript to frontend for live display
            await self.broadcast_to_clients({
                'type': 'live_transcript',
                'speaker': speaker,
                'text': transcript_data,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
            
            # Track transcript messages for note generation
            self.transcript_messages.append({
                'type': speaker.lower(),
                'text': transcript_data,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            })
            
            # Use the same logic as simple_live_churn.py
            # Check if speaker has changed for processing complete messages
            speaker_changed = self.churn_detector.current_speaker != speaker
            
            # If speaker changed, process the complete previous message
            if speaker_changed and self.churn_detector.current_speaker is not None:
                await self.process_complete_message()
            
            # Update current speaker and accumulate message (same as original)
            self.churn_detector.current_speaker = speaker
            
            if speaker.lower() == "agent":
                if speaker_changed:
                    self.churn_detector.current_agent_message = transcript_data
                else:
                    self.churn_detector.current_agent_message += " " + transcript_data
            elif speaker.lower() == "customer":
                if speaker_changed:
                    self.churn_detector.current_customer_message = transcript_data
                else:
                    self.churn_detector.current_customer_message += " " + transcript_data
                    
        except Exception as e:
            print(f"Error processing transcript: {e}")
    
    async def process_complete_message(self):
        """Process complete message when speaker changes"""
        try:
            if (self.churn_detector.current_speaker.lower() == "customer" and 
                self.churn_detector.current_customer_message.strip()):
                
                customer_text = self.churn_detector.current_customer_message.strip()
                agent_context = self.churn_detector.complete_agent_context
                
                print(f"\nCustomer: {customer_text}")
                
                # Don't send complete message - live transcript already sent
                
                # Process with churn scorer using the same method as simple_live_churn.py
                event = self.churn_detector.churn_scorer.process_customer_message(
                    customer_text, agent_context
                )
                
                # Display update with churn calculation (same as simple_live_churn.py)
                current_score = self.churn_detector.churn_scorer.get_current_score()
                if event.risk_delta != 0:
                    change_indicator = "+" if event.risk_delta > 0 else ""
                    print(f"Churn Score: {current_score}/100 (Risk Change: {change_indicator}{event.risk_delta:.1f})")
                else:
                    print(f"Churn Score: {current_score}/100 (No Change)")
                
                # Send churn update to frontend
                churn_update = {
                    'type': 'churn_update',
                    'current_score': current_score,
                    'risk_delta': event.risk_delta,
                    'sentiment': {
                        'score': event.sentiment_score,
                        'confidence': event.confidence
                    },
                    'emotion': {
                        'dominant_emotion': event.emotion_result['dominant_emotion'],
                        'dominant_score': event.emotion_result['dominant_score']
                    },
                    'patterns_detected': len(event.detected_patterns),
                    'detected_pattern_names': event.detected_patterns
                }
                
                print(f"Sending churn update to frontend: {churn_update}")
                await self.broadcast_to_clients(churn_update)
                
                # Get dynamic offers if churn score changed significantly
                if self.churn_detector.churn_scorer.should_trigger_offer_update():
                    offers = self.churn_detector.churn_scorer.get_offers_for_agent(customer_text)
                    if offers:
                        offers_update = {
                            'type': 'offers_update',
                            'offers': offers,
                            'triggered_by': 'churn_score_change',
                            'churn_score': current_score,
                            'risk_delta': event.risk_delta,
                            'timestamp': datetime.now().strftime('%H:%M:%S')
                        }
                        await self.broadcast_to_clients(offers_update)
                        print(f"Sent {len(offers)} dynamic offers to frontend")
                
                # Clear customer message
                self.churn_detector.current_customer_message = ""
                
            elif (self.churn_detector.current_speaker.lower() == "agent" and 
                  self.churn_detector.current_agent_message.strip()):
                
                # Store agent context and print it
                agent_text = self.churn_detector.current_agent_message.strip()
                print(f"\nAgent: {agent_text}")
                
                # Don't send complete message - live transcript already sent
                
                self.churn_detector.complete_agent_context = agent_text
                self.churn_detector.current_agent_message = ""
                
        except Exception as e:
            print(f"Error processing complete message: {e}")
    
    async def clear_conversation(self):
        """Clear conversation history and reset churn score"""
        try:
            if self.churn_detector:
                print("Clearing conversation data...")
                self.churn_detector.churn_scorer.reset_conversation()
                self.churn_detector.current_speaker = None
                self.churn_detector.current_agent_message = ""
                self.churn_detector.current_customer_message = ""
                self.churn_detector.complete_agent_context = ""
                self.transcript_messages = []  # Clear transcript messages
                
                await self.broadcast_to_clients({
                    'type': 'conversation_cleared',
                    'message': 'Conversation data cleared successfully'
                })
        except Exception as e:
            print(f"Error clearing conversation: {e}")
    
    async def handle_client_message(self, websocket, message):
        """Handle incoming messages from WebSocket clients"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'start_recording':
                clear_data = data.get('clear_data', False)
                await self.start_recording(clear_data)
            elif message_type == 'stop_recording':
                await self.stop_recording()
            elif message_type == 'clear_conversation':
                await self.clear_conversation()
            elif message_type == 'get_status':
                await websocket.send(json.dumps({
                    'type': 'status',
                    'is_recording': self.is_recording,
                    'models_loaded': self.churn_detector is not None
                }))
            elif message_type == 'generate_note':
                # Generate conversation note from recent messages
                if self.churn_detector:
                    transcript_messages = self.transcript_messages if hasattr(self, 'transcript_messages') else []
                    note = self.churn_detector.churn_scorer.generate_conversation_note(transcript_messages)
                    await websocket.send(json.dumps({
                        'type': 'note_generated',
                        'note': note,
                        'timestamp': datetime.now().strftime('%H:%M:%S')
                    }))
                else:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Churn detector not available'
                    }))
            elif message_type == 'generate_call_summary':
                # Generate comprehensive call summary
                if self.churn_detector:
                    transcript_messages = self.transcript_messages if hasattr(self, 'transcript_messages') else []
                    summary = self.churn_detector.churn_scorer.generate_call_summary(transcript_messages)
                    await websocket.send(json.dumps({
                        'type': 'call_summary_generated',
                        'summary': summary,
                        'timestamp': datetime.now().strftime('%H:%M:%S')
                    }))
                else:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Churn detector not available'
                    }))
            else:
                print(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            print("Invalid JSON received from client")
        except Exception as e:
            print(f"Error handling client message: {e}")
    
    async def handle_client(self, websocket):
        """Handle WebSocket client connection"""
        await self.register_client(websocket)
        try:
            async for message in websocket:
                await self.handle_client_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)

# Global server instance
server = WebSocketChurnServer()

async def main():
    """Main server function"""
    print("Initializing WebSocket Churn Detection Server...")
    
    # Initialize models
    await server.initialize_models()
    
    # Start WebSocket server
    print("Starting WebSocket server on ws://localhost:8765")
    start_server = websockets.serve(server.handle_client, "localhost", 8765)
    
    print("Server ready! Connect frontend to ws://localhost:8765")
    print("Waiting for client connections...")
    
    await start_server
    await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main()) 