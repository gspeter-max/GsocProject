import wave 
from google import genai
from google.genai import types
from typing import Tuple 

client = genai.Client( api_key = os.environ['GOOGLE_API_KEY'])

class Studio:
    def __init__( self , available_docs : str, person1 : Tuple[str,str] = ('peter','Enceladus'), \
        person2 : Tupl[str, str] = ('mark','algieba') ):
        self.available_docs = available_docs
        self.person1 = person1  # (in_role_name, voice_name) = ( peter, Enceladus)
        self.person2 = person2  # (in_role_name, voice_name) = ( mark , algieba )

    def generate_deep_deive_conversation( self,llm_generation_instruction : str= None):
        llm_generation_instruction = f'''
            You are simulating a vivid, human, highly intelligent, and natural-sounding conversation between two world-class expert humans — {self.person1[0]} and {self.person2[0]} — who are diving deep into a technical topic. They are having a back-and-forth discussion, teaching each other, brainstorming, asking tough questions, explaining details, correcting mistakes, and debating assumptions.The conversation should feel like a live deep-dive podcast, whiteboard session, or long-form expert discussion — rich with analogies, emotions, reasoning, false starts, corrections, equations, examples, and code if needed.You are NOT allowed to reference any "document", "this text", "the input", "the article", or anything similar. Pretend the speakers already know the material and are talking naturally — no meta-language.
            ### INPUT TOPIC:
            <<<
            {self.available_docs}
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
            {self.person1[0]}: ...
            {self.person2[0]} : ...

        '''  if llm_generation_instruction is None else llm_generation_instruction

        script = client.models.generate_content(
            model = 'gemini-2.5-flash', 
            contents = llm_generation_instruction,
        ).text

        return script

    def generate_conversation_speech( self,script: str ,output_audio_path : str ,\
                frame_rate = 24000,
                sample_width = 2,
                channels = 1
            ):

        generate_content_config = types.GenerateContentConfig(
            response_modalities = ['AUDIO'],
            speech_config= types.SpeechConfig(
                multi_speaker_voice_config= types.MultiSpeakerVoiceConfig(
                    speaker_voice_configs=[
                        types.SpeakerVoiceConfig(
                            speaker = self.person2[0],
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name = self.person1[1]
                                )
                            )
                        ),
                        types.SpeakerVoiceConfig(
                            speaker = self.person2[0],
                            voice_config = types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name = self.person2[1]
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

    def play_audio(self, audio_file_path):
        from IPython.display import Audio 
        Audio(audio_file_path, autopaly = True )
        

    def convert_to_mp3( self, wav_file_path, output_mp3_file_path):
        from pydub import AudioSegment 

        segment = AudioSegment.from_wav(wav_file_path)
        segment.export(output_mp3_file_path, format = 'mp3')


# url_data_class = url_data(query = 'what is llm ?' )
# url_docs = url_data_class.get_all_data(return_all_combined_resources = True)
# script = generate_deep_deive_conversation(url_docs)
# generate_conversation_speech(script ,output_audio_path = 'podcast')