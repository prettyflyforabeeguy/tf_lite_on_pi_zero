This example project using Python 3.7 is for runing a TensorFlow Lite model on a Raspberry Pi Zero W.
We'll be using a tensorflow model and example code created by Microsoft Lobe.

This same code will work on any Pi device as long as the correct .whl file is installed.

This model has 3 classifications:  HoneyBee, NoBee, and SomethingElse
Image 1 should detect as HoneyBee, Image 2 should detect as NoBee, and Image 3 should detect as SomethingElse

To get started:
1. Install the latest version of Rasbian OS
2. Download this code: git clone https://github.com/prettyflyforabeeguy/tf_lite_on_pi_zero.git
3. cd whl/armv6l
4. pip3 install tflite_runtime-2.3.1-cp37-cp37m-linux_armv6l.whl
5. cd /home/pi/tf_lite_on_pi_zero
6. python3 tf_example.py img/1.jpg

On the Pi Zero this will take about 1 minute to analzye the image and return the predicted results.

Alternatively you can install the the .whl on a pi3 or pi4
cd /whl/armv7l
pip3 install tflite_runtime-2.5.0-cp37-cp37m-linux_armv7l.whl

I believe TensorFlow no longer supports armv6l however, the most updated .whl files can be found here: 
https://www.tensorflow.org/lite/guide/python
