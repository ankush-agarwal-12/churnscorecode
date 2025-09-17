import asyncio
import json
import websockets
import threading
import queue
import os
from datetime import datetime
from typing import Dict, List
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
        
        # Initialize processing modes
        self.use_llm_processing = False
        self.use_hybrid_processing = False  # New hybrid mode
        self.llm_interval = 20
        self.llm_task = None
        self.last_llm_processing_time = None
        self.accumulated_messages = []
        self.churn_score_lock = asyncio.Lock()  # Thread safety for churn score updates
        
        # Parse environment variables for processing mode
        processing_mode = os.getenv('PROCESSING_MODE', 'rule').lower()
        if processing_mode == 'llm':
            self.use_llm_processing = True
            self.use_hybrid_processing = False
            print("🤖 LLM-only processing mode enabled")
        elif processing_mode == 'hybrid':
            self.use_llm_processing = False
            self.use_hybrid_processing = True
            print("🔄 Hybrid processing mode enabled (Rule-based + LLM)")
            print(f"   🧠 Rule-based: Churn scoring every message (offers disabled)")
            print(f"   🤖 LLM: Comprehensive analysis every {self.llm_interval} seconds (with offers)")
        else:
            self.use_llm_processing = False
            self.use_hybrid_processing = False
            print("🧠 Rule-based only processing mode enabled")
        
        # Legacy environment variable support
        llm_mode_env = os.getenv('USE_LLM_PROCESSING', 'false').lower()
        if llm_mode_env in ['true', '1', 'yes', 'on'] and processing_mode == 'rule':
            self.use_llm_processing = True
            print("🤖 LLM processing mode enabled via legacy environment variable")
        
        llm_interval_env = os.getenv('LLM_INTERVAL', '20')
        try:
            self.llm_interval = int(llm_interval_env)
            print(f"🤖 LLM processing interval set to {self.llm_interval} seconds")
        except ValueError:
            print(f"⚠️ Invalid LLM_INTERVAL value '{llm_interval_env}', using default 20 seconds")
        
    async def register_client(self, websocket):
        """Register a new WebSocket client"""
        self.websocket_clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.websocket_clients)}")
        
        # Set default initial customer profile for Megan Hazelwood (first customer in profile)
        if self.churn_detector:
            try:
                # Default customer profile data (matches Megan Hazelwood from customerProfiles.json)
                default_customer_data = {
                    "name": "Megan Hazelwood",
                    "currentMRC": "$200",
                    "previousMRC": "$180", 
                    "tenure": "18 months",
                    "currentPlan": "$200Data_TV_MOB_200MBPkg_2yr",
                    "currentProducts": "Internet, TV, Mobile"
                }
                default_customer_score = "50/100"  # Default to Megan Hazelwood's score
                
                await self._set_customer_profile(default_customer_data, default_customer_score)
                print(f"🎯 Set default customer profile for new client: {default_customer_data['name']} (MRC: {default_customer_data['currentMRC']}, Score: {default_customer_score})")
            except Exception as e:
                print(f"Error setting default customer profile: {e}")
        
    async def unregister_client(self, websocket):
        """Unregister a WebSocket client"""
        self.websocket_clients.discard(websocket)
        print(f"Client disconnected. Total clients: {len(self.websocket_clients)}")
        
    async def broadcast_to_clients(self, message):
        """Broadcast message to all connected clients"""
        if not self.websocket_clients:
            return
        
        # print(f"Broadcasting to {len(self.websocket_clients)} clients: {message.get('type', 'unknown')}")
        disconnected_clients = []
        
        for client in self.websocket_clients.copy():  # Use copy to avoid modification during iteration
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                print("Client disconnected during broadcast")
                disconnected_clients.append(client)
            except Exception as e:
                print(f"Error broadcasting to client: {e}")
                disconnected_clients.append(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            if client in self.websocket_clients:
                self.websocket_clients.remove(client)
                print(f"Removed disconnected client. Total clients: {len(self.websocket_clients)}")
    
    async def initialize_models(self):
        """Initialize all ML models (called once at startup)"""
        print("Initializing churn detection models...")
        self.churn_detector = SimpleLiveChurnDetector()
        
        # Configure churn detector based on processing mode
        if self.use_llm_processing:
            # LLM-only mode
            self.churn_detector.churn_scorer.use_llm_indicators = True
            self.churn_detector.churn_scorer.use_llm_offer_filtering = True
            print("🤖 Churn detector configured for LLM-only processing")
        elif self.use_hybrid_processing:
            # Hybrid mode - rule-based runs normally, LLM supplements
            self.churn_detector.churn_scorer.use_llm_indicators = False
            self.churn_detector.churn_scorer.use_llm_offer_filtering = False
            print("🔄 Churn detector configured for hybrid processing")
        else:
            # Rule-based only mode
            self.churn_detector.churn_scorer.use_llm_indicators = False
            self.churn_detector.churn_scorer.use_llm_offer_filtering = False
            print("🧠 Churn detector configured for rule-based only processing")
        
        # Send initialization complete message
        await self.broadcast_to_clients({
            'type': 'models_initialized',
            'message': 'All models loaded successfully',
            'llm_mode': self.use_llm_processing
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
            
            # Start LLM processing loop if LLM or hybrid mode is enabled
            if (self.use_llm_processing or self.use_hybrid_processing):
                if not self.llm_task or self.llm_task.done():
                    self.llm_task = asyncio.create_task(self.llm_processing_loop())
                    mode_name = "LLM" if self.use_llm_processing else "Hybrid"
                    print(f"🤖 Started {mode_name} processing loop")
            
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
            
            # Cancel LLM processing task
            if self.llm_task:
                self.llm_task.cancel()
                self.llm_task = None
                print("🤖 Stopped LLM processing loop")
            
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
                            # print(f"Audio frames captured: {self._audio_frame_count}")
                            pass
                    else:
                        self._audio_frame_count = 1
                        # print("Audio capture started - first frame captured")
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
                        # print(f"Audio frames sent to Deepgram: {sent_frames}")
                        pass
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
            
            # print(f"Processing transcript - Speaker: {speaker}, Text: {transcript_data}")
            
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
            
            # Accumulate messages for LLM processing if LLM or hybrid mode is enabled
            if self.use_llm_processing or self.use_hybrid_processing:
                # Merge consecutive messages from same speaker before accumulating
                if self.accumulated_messages and self.accumulated_messages[-1]['speaker'] == speaker:
                    # Same speaker continuing - merge the messages
                    self.accumulated_messages[-1]['text'] += ' ' + transcript_data
                    self.accumulated_messages[-1]['timestamp'] = datetime.now()
                else:
                    # New speaker or first message - add as new message
                    self.accumulated_messages.append({
                        'speaker': speaker,
                        'text': transcript_data,
                        'timestamp': datetime.now()
                    })
                
                # In LLM-only mode, skip rule-based processing
                if self.use_llm_processing:
                    return  # Skip rule-based processing
            
            # Use the same logic as simple_live_churn.py (only for rule-based mode)
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
                
                # print(f"\nCustomer: {customer_text}")
                
                # Don't send complete message - live transcript already sent
                
                # Process churn scoring asynchronously to avoid blocking audio
                asyncio.create_task(self._process_churn_scoring_async(customer_text, agent_context))
                
                # Clear customer message
                self.churn_detector.current_customer_message = ""
                
            elif (self.churn_detector.current_speaker.lower() == "agent" and 
                  self.churn_detector.current_agent_message.strip()):
                
                # Store agent context and print it
                agent_text = self.churn_detector.current_agent_message.strip()
                # print(f"\nAgent: {agent_text}")
                
                # Don't send complete message - live transcript already sent
                
                self.churn_detector.complete_agent_context = agent_text
                self.churn_detector.current_agent_message = ""
                
        except Exception as e:
            print(f"Error processing complete message: {e}")
    
    async def llm_processing_loop(self):
        """Process accumulated messages with LLM every 20 seconds"""
        try:
            while (self.use_llm_processing or self.use_hybrid_processing) and self.is_recording:
                await asyncio.sleep(self.llm_interval)
                
                if self.accumulated_messages:
                    mode_name = "LLM-only" if self.use_llm_processing else "Hybrid"
                    print(f"🤖 {mode_name} mode: Processing {len(self.accumulated_messages)} accumulated messages with LLM")
                    await self.process_with_llm()
                else:
                    mode_name = "LLM-only" if self.use_llm_processing else "Hybrid"
                    print(f"🤖 {mode_name} mode: No new messages to process with LLM")
                    
        except asyncio.CancelledError:
            print("🤖 LLM processing loop cancelled")
        except Exception as e:
            print(f"Error in LLM processing loop: {e}")
    
    async def process_with_llm(self):
        """Process accumulated messages with LLM analysis"""
        if not self.accumulated_messages:
            return
        
        try:
            # Build conversation context - use last 6-8 messages for 3-4 conversation turns
            context_messages = []
            current_message = ""
            current_speaker = ""
            
            # Limit to recent conversation for better LLM context (last 8 messages max)
            recent_messages = self.accumulated_messages[-8:] if len(self.accumulated_messages) > 8 else self.accumulated_messages
            
            if len(recent_messages) >= 1:
                last_msg = recent_messages[-1]
                current_message = last_msg['text']
                current_speaker = last_msg['speaker']
                
                # Build context from previous messages (excluding the current one) - aim for 3-4 turns
                context_msgs = recent_messages[:-1]  # All except the last one
                for msg in context_msgs:
                    context_messages.append(f"{msg['speaker']}: {msg['text']}")
            
            if not current_message:
                print("⚠️ No current message to analyze")
                return
            
            # If agent was speaking, analyze the last customer message instead
            if current_speaker == "Agent" and len(recent_messages) >= 2:
                # Find the last customer message in recent messages
                for i in range(len(recent_messages) - 2, -1, -1):
                    if recent_messages[i]['speaker'] == "Customer":
                        current_message = recent_messages[i]['text']
                        current_speaker = "Customer"
                        print(f"   → Agent was speaking, switched to analyze customer message: '{current_message[:50]}{'...' if len(current_message) > 50 else ''}'")
                        
                        # Rebuild context excluding this customer message
                        context_messages = []
                        for j, msg in enumerate(recent_messages):
                            if j != i:  # Skip the customer message we're analyzing
                                context_messages.append(f"{msg['speaker']}: {msg['text']}")
                        break
            
            print(f"🤖 Processing {len(self.accumulated_messages)} accumulated messages with LLM")
            print(f"🤖 LLM Processing ({len(recent_messages)} recent messages):")
            print(f"   Analyzing {current_speaker} message: '{current_message[:50]}{'...' if len(current_message) > 50 else ''}'")
            if context_messages:
                print(f"   Context ({len(context_messages)} messages for 3-4 conversation turns):")
                for i, ctx_msg in enumerate(context_messages, 1):  # Show ALL context messages
                    print(f"     {i}. {ctx_msg[:80]}{'...' if len(ctx_msg) > 80 else ''}")
            else:
                print(f"   No context messages")
            
            # Get comprehensive analysis from LLM
            analysis = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.churn_detector.churn_scorer.llm_extractor.get_comprehensive_analysis,
                current_message,
                "\n".join(context_messages)
            )
            
            if "error" in analysis:
                print(f"❌ LLM analysis failed: {analysis['error']}")
                return
            
            # Process churn analysis
            async with self.churn_score_lock:
                event = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._process_churn_from_llm_analysis,
                    analysis,
                    current_message,
                    "\n".join(context_messages)
                )
            
            current_score = self.churn_detector.churn_scorer.current_score
            
            print(f"🤖 Results:")
            print(f"   Sentiment: {analysis.get('sentiment', 'unknown')}, Emotion: {analysis.get('emotion', 'unknown')}")
            print(f"   Risk Patterns: {analysis.get('risk_patterns', [])}")
            print(f"   Final Churn Score: {current_score}/100")
            
            if event.risk_delta != 0:
                # Send churn score update
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
                    'detected_pattern_names': event.detected_patterns,
                    'processing_mode': 'hybrid' if self.use_hybrid_processing else 'llm'
                }
                await self.broadcast_to_clients(churn_update)
            
            # Process offers - always send in hybrid/LLM mode
            if self.use_llm_processing:
                # LLM-only mode: use LLM for offer filtering
                llm_offers = self.churn_detector.churn_scorer.get_offers_for_agent_with_analysis(current_message, analysis)
                if llm_offers:
                    print(f"🤖 Sending {len(llm_offers)} offers (including rejected) to frontend")
                    
                    offers_update = {
                        'type': 'offers_update',
                        'offers': llm_offers,  # This includes both accepted and rejected offers
                        'triggered_by': 'llm_churn_analysis',
                        'churn_score': current_score,
                        'risk_delta': event.risk_delta,
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'processing_mode': 'llm'
                    }
                    await self.broadcast_to_clients(offers_update)
            elif self.use_hybrid_processing:
                # Hybrid mode: use LLM-only for offer filtering, not rule-based
                llm_offers = self.churn_detector.churn_scorer.get_offers_for_agent_with_analysis(current_message, analysis)
                if llm_offers:
                    print(f"🔄 Hybrid mode: Sending {len(llm_offers)} LLM-filtered offers (no rule-based filtering)")
                    
                    offers_update = {
                        'type': 'offers_update',
                        'offers': llm_offers,
                        'triggered_by': 'hybrid_llm_analysis',
                        'churn_score': current_score,
                        'risk_delta': event.risk_delta,
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'processing_mode': 'hybrid',
                        'llm_insights': {
                            'sentiment': analysis.get('sentiment', 'unknown'),
                            'emotion': analysis.get('emotion', 'unknown'),
                            'risk_patterns': analysis.get('risk_patterns', [])
                        }
                    }
                    await self.broadcast_to_clients(offers_update)
            
            # Clear processed messages
            self.accumulated_messages = []
            
        except Exception as e:
            print(f"❌ Error in LLM processing: {e}")
            import traceback
            traceback.print_exc()
    
    def _process_churn_from_llm_analysis(self, llm_analysis: dict, customer_message: str, context: str):
        """Process churn scoring using existing LLM analysis results"""
        # Extract LLM results
        sentiment = llm_analysis.get("sentiment", "neutral")
        emotion = llm_analysis.get("emotion", "neutral")
        risk_patterns = llm_analysis.get("risk_patterns", [])
        
        print(f"🤖 Results:")
        print(f"   Sentiment: {sentiment}, Emotion: {emotion}")
        print(f"   Risk Patterns: {risk_patterns}")
        
        # Convert LLM sentiment to numerical score
        sentiment_score = self.churn_detector.churn_scorer._convert_llm_sentiment_to_score(sentiment)
        
        # Create emotion result in expected format
        emotion_result = {
            'dominant_emotion': emotion,
            'dominant_score': 0.8  # Default confidence for LLM emotion
        }
        
        # Calculate pattern risk
        pattern_risk = self.churn_detector.churn_scorer._calculate_llm_pattern_risk(risk_patterns)
        
        # Calculate sentiment risk (same logic as original)
        sentiment_risk = 0.0
        if sentiment_score < -0.6:
            sentiment_risk = 30.0
        elif sentiment_score < -0.3:
            sentiment_risk = 20.0
        elif sentiment_score < -0.1:
            sentiment_risk = 10.0
        elif sentiment_score > 0.36:
            sentiment_risk = -20.0
        elif sentiment_score > 0.25:
            sentiment_risk = -10.0
        
        print(f"Sentiment Risk: {sentiment_risk:.1f} (based on sentiment score {sentiment_score:.3f})")
        
        # Calculate emotion risk
        print("Emotion Risk Analysis:")
        emotion_risk = self.churn_detector.churn_scorer.calculate_emotion_risk(emotion_result)
        print(f"Emotion Risk: {emotion_risk:.1f}")
        
        # Calculate total risk
        total_risk_delta = pattern_risk + sentiment_risk + emotion_risk
        print(f"Total Risk Delta: {pattern_risk:.1f} + {sentiment_risk:.1f} + {emotion_risk:.1f} = {total_risk_delta:.1f}")
        
        # Update cumulative score using latest score (thread-safe)
        previous_score = self.churn_detector.churn_scorer.current_score  # Get latest score
        alpha = self.churn_detector.churn_scorer.alpha
        new_score = (1 - alpha) * previous_score + alpha * (previous_score + total_risk_delta)
        new_score = max(0.0, min(100.0, new_score))
        
        print(f"Score Update: ({1-alpha:.1f} × {previous_score:.1f}) + ({alpha:.1f} × ({previous_score:.1f} + {total_risk_delta:.1f})) = {new_score:.1f}")
        print(f"🔄 Hybrid: LLM updating score from {previous_score:.1f} to {new_score:.1f}")
        
        self.churn_detector.churn_scorer.current_score = new_score
        
        # Create event record
        from simple_churn_scorer import ChurnEvent
        event = ChurnEvent(
            timestamp=datetime.now(),
            speaker="Customer",
            text=customer_message,
            agent_context=context,
            sentiment_score=sentiment_score,
            emotion_result=emotion_result,
            risk_delta=total_risk_delta,
            cumulative_score=new_score,
            confidence=0.8,
            detected_patterns=risk_patterns
        )
        
        # Store in history
        self.churn_detector.churn_scorer.conversation_history.append(event)
        self.churn_detector.churn_scorer.risk_events.append(event)
        
        return event
     
    def _generate_llm_rejection_reason(self, offer: Dict, sentiment: str, emotion: str, risk_patterns: List[str], llm_analysis: dict) -> str:
        """Generate specific rejection reason for offers using LLM analysis"""
        # Extract offer indicators for enhanced reasoning
        offer_indicators = llm_analysis.get('offer_indicators', {})
        
        # Use the enhanced rejection reason method from churn scorer
        if hasattr(self.churn_detector.churn_scorer, '_generate_enhanced_llm_rejection_reason'):
            return self.churn_detector.churn_scorer._generate_enhanced_llm_rejection_reason(
                offer, sentiment, emotion, risk_patterns, offer_indicators
            )
        
        # Fallback to simple reasons if enhanced method not available
        reasons = []
        
        # Check budget/price sensitivity
        if sentiment in ['negative', 'very_negative'] and 'billing_complaint' in risk_patterns:
            if offer['price_delta'] > 0:
                reasons.append(f"Price increase conflicts with billing complaints")
            else:
                reasons.append("Not aligned with budget concerns")
        
        # Check service usage patterns
        offer_types = offer.get('product_types', [])
        if 'TV' in offer_types:
            tv_indicators = offer_indicators.get('service_usage', {})
            if tv_indicators.get('tv_usage') == 'low':
                reasons.append("Customer barely watches TV")
        
        # Check emotional state
        if emotion in ['anger', 'frustration'] and offer['category'] in ['upgrade', 'premium']:
            reasons.append(f"Customer {emotion} - premium not suitable")
        
        # Default reason if no specific reason found
        if not reasons:
            if sentiment in ['negative', 'very_negative']:
                reasons.append(f"Not suitable for {sentiment} sentiment")
            else:
                reasons.append("Lower priority based on analysis")
        
        return "; ".join(reasons)
     
    def _convert_llm_offers_to_frontend_format(self, llm_offers: list) -> list:
        """Convert LLM filtered offers to frontend-compatible format"""
        simplified_offers = []
        for offer in llm_offers:
            simplified_offers.append({
                'id': offer.get('offer_id', ''),
                'title': offer.get('title', ''),
                'description': offer.get('description', ''),
                'value': f"${offer.get('price_delta', 0)}/month",
                'urgency': offer.get('urgency', 'Standard'),
                'category': offer.get('category', 'bundle'),
                'relevance': 80,  # Default relevance for LLM offers
                'type': offer.get('category', 'bundle'),
                'price_delta': offer.get('price_delta', 0),
                'retention_offer': offer.get('retention_offer', False),
                'accepted': True,  # LLM filtered offers are accepted
                'rejection_reason': None,
                'llm_filtered': True
            })
        return simplified_offers
 
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
                
                # Clear LLM processing data
                if self.use_llm_processing:
                    self.accumulated_messages = []
                    self.last_llm_processing_time = datetime.now()
                    print("🤖 Cleared LLM accumulated messages")
                
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
                    'models_loaded': self.churn_detector is not None,
                    'llm_processing': self.use_llm_processing,
                    'llm_interval': self.llm_interval
                }))
            elif message_type == 'set_llm_mode':
                mode = data.get('mode', 'rule')  # 'rule', 'llm', or 'hybrid'
                await self.set_processing_mode(mode)
            elif message_type == 'get_processing_mode':
                current_mode = self.get_current_processing_mode()
                response = {
                    'type': 'processing_mode_status',
                    'mode': current_mode
                }
                await websocket.send(json.dumps(response))
            elif message_type == 'generate_note':
                # Generate conversation note from recent messages (non-blocking)
                if self.churn_detector:
                    asyncio.create_task(self._generate_note_async(websocket))
                else:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Churn detector not available'
                    }))
            elif message_type == 'generate_call_summary':
                # Generate comprehensive call summary (non-blocking)
                if self.churn_detector:
                    asyncio.create_task(self._generate_call_summary_async(websocket))
                else:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Churn detector not available'
                    }))
            elif message_type == 'set_customer_baseline':
                # Set baseline churn score for selected customer
                if self.churn_detector:
                    churn_risk_score = data.get('churn_risk_score', '50/100')
                    await self._set_customer_baseline(churn_risk_score)
                    await websocket.send(json.dumps({
                        'type': 'customer_baseline_set',
                        'baseline_score': self.churn_detector.churn_scorer.baseline_score,
                        'message': f'Baseline score set to {self.churn_detector.churn_scorer.baseline_score}/100'
                    }))
                else:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Churn detector not available'
                    }))
            elif message_type == 'set_customer_profile':
                # Set customer profile data (including MRC for offer filtering)
                if self.churn_detector:
                    customer_data = data.get('customer_data', {})
                    churn_risk_score = data.get('churn_risk_score', '50/100')
                    await self._set_customer_profile(customer_data, churn_risk_score)
                    await websocket.send(json.dumps({
                        'type': 'customer_profile_set',
                        'baseline_score': self.churn_detector.churn_scorer.baseline_score,
                        'customer_name': customer_data.get('name', 'Unknown'),
                        'message': f'Customer profile updated: {customer_data.get("name", "Unknown")}'
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
                try:
                    await self.handle_client_message(websocket, message)
                except Exception as e:
                    print(f"Error handling client message: {e}")
                    # Don't disconnect on message handling errors
                    try:
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'message': f'Error processing message: {str(e)}'
                        }))
                    except:
                        break  # Connection is broken, exit
        except websockets.exceptions.ConnectionClosed:
            print("Client connection closed normally")
        except Exception as e:
            print(f"Unexpected error in client handler: {e}")
        finally:
            await self.unregister_client(websocket)

    async def _generate_note_async(self, websocket):
        """Generate conversation note asynchronously without blocking audio processing"""
        try:
            transcript_messages = self.transcript_messages if hasattr(self, 'transcript_messages') else []
            
            # Run the blocking operation in a thread pool
            loop = asyncio.get_event_loop()
            note = await loop.run_in_executor(
                None, 
                self.churn_detector.churn_scorer.generate_conversation_note, 
                transcript_messages
            )
            
            await websocket.send(json.dumps({
                'type': 'note_generated',
                'note': note,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }))
        except Exception as e:
            print(f"Error generating note: {e}")
            await websocket.send(json.dumps({
                'type': 'error',
                'message': f'Error generating note: {str(e)}'
            }))

    async def _generate_call_summary_async(self, websocket):
        """Generate call summary asynchronously without blocking audio processing"""
        try:
            transcript_messages = self.transcript_messages if hasattr(self, 'transcript_messages') else []
            
            # Run the blocking operation in a thread pool
            loop = asyncio.get_event_loop()
            summary = await loop.run_in_executor(
                None, 
                self.churn_detector.churn_scorer.generate_call_summary, 
                transcript_messages
            )
            
            await websocket.send(json.dumps({
                'type': 'call_summary_generated',
                'summary': summary,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }))
        except Exception as e:
            print(f"Error generating call summary: {e}")
            await websocket.send(json.dumps({
                'type': 'error',
                'message': f'Error generating call summary: {str(e)}'
            }))

    async def _set_customer_baseline(self, churn_risk_score: str):
        """Set the customer baseline score from the churn risk score string"""
        try:
            # Extract numeric score from "75/100" format
            if '/' in churn_risk_score:
                score_str = churn_risk_score.split('/')[0]
            else:
                score_str = churn_risk_score
            
            score = float(score_str)
            
            # Set the baseline score in the churn scorer
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.churn_detector.churn_scorer.set_baseline_score,
                score
            )
            
            # Broadcast the new baseline to all clients
            await self.broadcast_to_clients({
                'type': 'baseline_score_updated',
                'baseline_score': score,
                'current_score': self.churn_detector.churn_scorer.current_score,
                'message': f'Customer baseline score updated to {score}/100'
            })
            
        except (ValueError, AttributeError) as e:
            print(f"Error setting customer baseline: {e}")
            raise

    async def _set_customer_profile(self, customer_data: dict, churn_risk_score: str):
        """Set the complete customer profile including baseline score and profile data"""
        try:
            # Set baseline score first
            await self._set_customer_baseline(churn_risk_score)
            
            # Set customer profile data in the churn scorer (for LLM extractor)
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.churn_detector.churn_scorer.set_customer_profile_data,
                customer_data
            )
            
            # Broadcast the complete profile update to all clients
            await self.broadcast_to_clients({
                'type': 'customer_profile_updated',
                'customer_name': customer_data.get('name', 'Unknown'),
                'baseline_score': self.churn_detector.churn_scorer.baseline_score,
                'current_score': self.churn_detector.churn_scorer.current_score,
                'current_mrc': customer_data.get('currentMRC', 'Unknown'),
                'message': f'Complete customer profile updated: {customer_data.get("name", "Unknown")}'
            })
            
        except Exception as e:
            print(f"Error setting customer profile: {e}")
            raise

    async def _process_churn_scoring_async(self, customer_text: str, agent_context: str):
        """Process churn scoring asynchronously to avoid blocking audio processing"""
        try:
            # Use thread-safe processing in hybrid mode
            if self.use_hybrid_processing:
                async with self.churn_score_lock:
                    event = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        self.churn_detector.churn_scorer.process_customer_message,
                        customer_text,
                        agent_context
                    )
            else:
                # No lock needed for rule-based only or LLM-only modes
                event = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    self.churn_detector.churn_scorer.process_customer_message,
                    customer_text,
                    agent_context
                )
            
            # Display update with churn calculation (same as simple_live_churn.py)
            current_score = self.churn_detector.churn_scorer.get_current_score()
            if event.risk_delta != 0:
                change_indicator = "+" if event.risk_delta > 0 else ""
                # print(f"Churn Score: {current_score}/100 (Risk Change: {change_indicator}{event.risk_delta:.1f})")
            else:
                # print(f"Churn Score: {current_score}/100 (No Change)")
                pass
            
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
            
            # print(f"Sending churn update to frontend: {churn_update}")
            await self.broadcast_to_clients(churn_update)
            
            # Get dynamic offers if churn score changed significantly
            # In hybrid mode, skip rule-based offer filtering - only LLM should trigger offers
            if not self.use_hybrid_processing and self.churn_detector.churn_scorer.should_trigger_offer_update(churn_delta_threshold=3.5):
                # print(f"✅ Triggering offer update due to significant churn change: {event.risk_delta}")
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
                    # print(f"📤 Sent {len(offers)} dynamic offers to frontend")
                else:
                    # print("⚠️ No offers returned from get_offers_for_agent")
                    pass
            elif self.use_hybrid_processing:
                print(f"🔄 Hybrid mode: Skipping rule-based offer filtering (churn delta: {event.risk_delta:.1f}) - waiting for LLM")
            else:
                # print(f"❌ Offer update not triggered - Risk delta {event.risk_delta} below threshold of 4.0")
                pass
                
        except Exception as e:
            print(f"Error processing churn scoring: {e}")

    async def set_processing_mode(self, mode: str):
        """Set the processing mode: 'rule', 'llm', or 'hybrid'"""
        old_mode = self.get_current_processing_mode()
        
        if mode.lower() == 'llm':
            self.use_llm_processing = True
            self.use_hybrid_processing = False
            self.churn_detector.churn_scorer.use_llm_indicators = True
            self.churn_detector.churn_scorer.use_llm_offer_filtering = True
            print(f"🤖 Switched to LLM-only processing mode")
        elif mode.lower() == 'hybrid':
            self.use_llm_processing = False
            self.use_hybrid_processing = True
            self.churn_detector.churn_scorer.use_llm_indicators = False
            self.churn_detector.churn_scorer.use_llm_offer_filtering = False
            print(f"🔄 Switched to Hybrid processing mode")
        else:  # 'rule' or any other value defaults to rule-based
            self.use_llm_processing = False
            self.use_hybrid_processing = False
            self.churn_detector.churn_scorer.use_llm_indicators = False
            self.churn_detector.churn_scorer.use_llm_offer_filtering = False
            print(f"🧠 Switched to Rule-based only processing mode")
        
        # Start or stop LLM processing loop as needed
        if (self.use_llm_processing or self.use_hybrid_processing):
            if not self.llm_task or self.llm_task.done():
                self.llm_task = asyncio.create_task(self.llm_processing_loop())
                print(f"🤖 Started LLM processing loop for {mode} mode")
        else:
            if self.llm_task and not self.llm_task.done():
                self.llm_task.cancel()
                print(f"⏹️ Stopped LLM processing loop")
        
        # Notify all clients of the mode change
        new_mode = self.get_current_processing_mode()
        await self.broadcast_to_clients({
            'type': 'processing_mode_changed',
            'old_mode': old_mode,
            'new_mode': new_mode,
            'message': f"Processing mode changed from {old_mode} to {new_mode}"
        })

    def get_current_processing_mode(self) -> str:
        """Get the current processing mode as a string."""
        if self.use_llm_processing:
            return "LLM-only"
        elif self.use_hybrid_processing:
            return "Hybrid (Rule-based + LLM)"
        else:
            return "Rule-based only"

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