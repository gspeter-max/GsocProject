import wave 
from google import genai

client = genai.Client( api_key = os.environ['GOOGLE_API_KEY'])

def generate_deep_deive_conversation( available_docs : str ):
    llm_content_generation = f'''
        You are simulating a vivid, human, highly intelligent, and natural-sounding conversation between two world-class expert humans — Person A and Person B — who are diving deep into a technical topic. They are having a back-and-forth discussion, teaching each other, brainstorming, asking tough questions, explaining details, correcting mistakes, and debating assumptions.The conversation should feel like a live deep-dive podcast, whiteboard session, or long-form expert discussion — rich with analogies, emotions, reasoning, false starts, corrections, equations, examples, and code if needed.You are NOT allowed to reference any "document", "this text", "the input", "the article", or anything similar. Pretend the speakers already know the material and are talking naturally — no meta-language.
        ### INPUT TOPIC:
        <<<
        [Insert your document or topic content here, such as explanation of a model, tutorial, research, whitepaper, guide, etc.]
        >>>

        ### RULES FOR GENERATING THE CONVERSATION:

        - DO NOT say “the document says” or anything like that.
        - Instead, generate natural-sounding speech where the two speakers explore the ideas as if they already understand them.
        - The tone should sound like two PhDs, senior engineers, or brilliant thinkers.
        - Include real questions like:
            - “Wait, how does this actually work under the hood?”
            - “I used to think X, but now I realize Y — does that make sense?”
            - “Let me write this on the whiteboard…” (include code or math here)
        - Include analogies, breakdowns, and step-by-step explanations.
        - Show deep reasoning, curiosity, and collaborative thinking.
        - Include at least 50 exchanges. Make it long, like a full notebook session or podcast episode.

        ### FORMAT:
        peter: ...
        mark : ...

    '''

    script = client.models.generate_content(
        model = 'gemini-2.5-flash', 
        contents = llm_content_generation,
    ).text

    return script

def generate_conversation_speech( script: str ,output_audio_path : str ,\
            frame_rate = 24000,
            sample_width = 2,
            channels = 1
        ):

    generate_content_config = types.GenerateContentConfig(
        response_modelities = ['AUDIO'],
        speech_config= types.SpeechConfig(
            multi_speaker_voice_config= types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=[
                    types.SpeakerVoiceConfig(
                        speaker = 'peter',
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name = 'sulafat'
                            )
                        )
                    ),
                    types.SpeakerVoiceConfig(
                        speaker = 'mark',
                        voice_config = types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name = 'Achernar'
                            )
                        )
                    )
                ]
            )
        )
    )

    response = client.models.generate_content(
        model = 'gemini-2.5-flash-preview-tts',
        contents = 'say this ' + 'ok start' + script,
        config = generate_content_config
    )

    blob = response.candidates[0].content.parts[0].inline_data
    file_path = output_audio_path + '.wav'

    with wave.open( file_path ,'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(frame_rate)
        wf.writeframes(blob.data)

def play_audio(audio_file_path):
    from IPython.display import Audio 
    Audio(audio_file_path, autopaly = True )
    
