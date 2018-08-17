#for linux commands
import os

#for clearing the memory used by the previous execution
import sys

#for sleep
from time import sleep

#for storing history
from datetime import datetime
import shutil


#continuous loop for constantly processing images
while True:
        #remove the last image captured
        #os.system("rm ./pi_images/example.jpg")

        #check if there is any new image recieved from the remote device
        while (os.path.isfile("./pi_images/example.jpg")==False):

            #if not sleep for 5 seconds
            sleep(5)

            #again check for any new images captured by the remote device
            #os.system("scp pi@192.168.43.7:example.jpg ./pi_images/")

        #run the neural network for the object detection in the given image
        os.chdir("./object_detection")
        import object_detection.object_detection_Images_Only_Resnet
        os.chdir("../Places")
     
        #run the neural network for scene classification for the given image
        import Places.run_placesCNN_basic
        os.chdir("../")

        #for Google's Text-To-Speech engine (A more natural sounding speech)
        from gtts import gTTS

        #oneBigString stores the result obtained after running the neural network
        oneBigString = open("record.txt",'r').read()
        oneBigString = oneBigString.replace("_", ":")
        print("inside12")	
		  
        #run the gTTS modul
        tts = gTTS(text=oneBigString, lang="en")
        print("inside1")

        #save the audio file generated
        tts.save("record.mp3")

        #display the results onto the server screen
        print(oneBigString)

        #transfer the recorded mp3 file to the remote device
        #os.system("scp record.mp3 pi@192.168.43.7:.")

        #make a record of this processing by creating a folder of the current time and saving the results in that
        newFolder = "./history/"+str(datetime.now())

        #make new folder with the current timestamp
        os.mkdir(newFolder)

        #copy files for future reference
        shutil.copy('./pi_images/example.jpg', newFolder) #for image copying

        shutil.copy("record.txt", newFolder) #for result copying

        shutil.copy("record.mp3", newFolder) #for audio file copying

        #refresh or reset the memory (GPU)
        os.execl(sys.executable, sys.executable, *sys.argv)

        #sleep for sometime
        sleep(10)

