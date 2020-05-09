import streamlit as st
import cv2
from PIL import Image,ImageEnhance
import numpy as np
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
fullbody_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

@st.cache
def load_image(img):
	im = Image.open(img)
	return im

@st.cache
def detect_faces(our_image):
	new_img = np.array(our_image)
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#Detect Face
	faces = face_cascade.detectMultiScale(gray,1.1,4)
	# Draw Rectangle
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	return img,faces

def detect_eyes(our_image):
	new_img = np.array(our_image)
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	eyes = eye_cascade.detectMultiScale(gray,1.3,5)
	for (ex,ey,ew,eh) in eyes:
		cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	return img

def detetct_smiles(our_image):
	new_img = np.array(our_image)
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	smiles = smile_cascade.detectMultiScale(gray,1.1,4)
	for (x,y,w,h) in smiles:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	return img

def cartoonize_img(our_image):
	new_img = np.array(our_image)
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	gray = cv2.medianBlur(gray,5)
	edges = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,blockSize=9,C=2)
	color = cv2.bilateralFilter(img,9,300,300)
	cartoon = cv2.bitwise_and(color, color, mask=edges)

	return cartoon

def cannize_image(our_image):
	new_img = np.array(our_image)
	img = cv2.cvtColor(new_img,1)
	img=cv2.GaussianBlur(img,(11,11),0)
	canny = cv2.Canny(img,100,150)
	return canny


def main():
	"""
	Face Detection Application
	"""
	st.title("Face Detection Application using OpenCV")
	activities = ["Detection","About"]
	choice = st.sidebar.selectbox("Select Activity",activities)

	if choice == "Detection":
		st.subheader("Face Detection")

		image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])

		if image_file is not None:
			our_image = Image.open(image_file).convert('RGB')
			st.text("Original Image")
			st.image(our_image)

		enhance_type = st.sidebar.radio("Enhance eType:",["Original","Gray-Scale","Contrast","Brightness","Blurring"])
		if enhance_type == 'Gray-Scale':
			new_img = np.array(our_image)
			img = cv2.cvtColor(new_img,1)
			img_output = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			st.image(img_output)

		if enhance_type == 'Contrast':
			c_rate = st.sidebar.slider("Contrast",0.5,3.5)
			enhancer = ImageEnhance.Contrast(our_image)
			img_output = enhancer.enhance(c_rate)
			st.image(img_output)

		if enhance_type == 'Brightness':
			c_rate = st.sidebar.slider("Brightness",0.5,3.5)
			enhancer = ImageEnhance.Brightness(our_image)
			img_output = enhancer.enhance(c_rate)
			st.image(img_output)

		if enhance_type == 'Blurring':
			blur_rate = st.sidebar.slider("Blur Rate",0.5,3.5)
			new_img = np.array(our_image)
			img = cv2.cvtColor(new_img,1)
			img_output = cv2.GaussianBlur(img,(11,11),blur_rate)
			st.image(img_output)

		# else:
		# 	st.image(our_image,width=300)

		task = ["Faces","Smiles","Eyes","Cannize","Cartoonize"]
		feature_choice = st.sidebar.selectbox("Find Features",task)

		if st.button("Process"):

			if feature_choice == 'Faces':
				result_img,result_faces = detect_faces(our_image)
				st.image(result_img)

				st.success("Found {} faces".format(len(result_faces)))

			elif feature_choice == 'Smiles':
				result_img = detetct_smiles(our_image)
				st.image(result_img)

			elif feature_choice == 'Eyes':
				result_img = detect_eyes(our_image)
				st.image(result_img)

			elif feature_choice == 'Cannize':
				result_img = cannize_image(our_image)
				st.image(result_img)

			elif feature_choice == 'Cartoonize':
				result_img = cartoonize_img(our_image)
				st.image(result_img)
							



	elif choice == "About":
		st.subheader("About")


if __name__ == '__main__':
	main()
