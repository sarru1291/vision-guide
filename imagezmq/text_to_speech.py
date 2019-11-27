import subprocess

# text to speech function
def execute_unix(inputcommand):
	p = subprocess.Popen(inputcommand, stdout=subprocess.PIPE, shell=True)
	(output, err) = p.communicate()
	return output
	
def text_to_speech(text):
	b='espeak -w /home/pi/Desktop/vision_guide/imagezmq/audio_output.wav "%s" 2>>/dev/null' % text
	c='espeak -ven+f3 -k5 -s150 --punct="<characters>" "%s" 2>>/dev/null' % text
	execute_unix(b)
	execute_unix(c)
