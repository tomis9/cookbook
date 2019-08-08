import speech_recognition as sr

sr.__version__

r = sr.Recognizer()

harvard = sr.AudioFile('harvard.wav')
with harvard as source:
    audio = r.record(source)


r.recognize_google(audio)

jackhammer = sr.AudioFile('jackhammer.wav')
with jackhammer as source:
    audio = r.record(source)

r.recognize_google(audio)


# sudo apt-get install portaudio19-dev
