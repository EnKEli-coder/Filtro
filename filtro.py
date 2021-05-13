import cv2
import imutils

# Videostreaming o video de entrada
video = cv2.VideoCapture(1,cv2.CAP_DSHOW)

# Lectura de la imagen a incrustar en el video
#image = cv2.imread('gorro_navidad.png', cv2.IMREAD_UNCHANGED)
#image = cv2.imread('cap.png', cv2.IMREAD_UNCHANGED)
image = cv2.imread('helmet.png', cv2.IMREAD_UNCHANGED)

# Detector de rostros
detector_rostros = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:

	ret, frame = video.read()
	if ret == False: break
	frame = imutils.resize(frame, width=640)

	# Detección de rostros
	rostros = detector_rostros.detectMultiScale(frame, 1.3, 5)

	for (x, y, w, h) in rostros:

		# Ajustamos la imagen al rostro detectado
		resized_image = imutils.resize(image, width=w)
		filas_image = resized_image.shape[0]
		col_image = w

	
		# Determinamos a que altura aparecera la imagen
		porcion_alto = filas_image 

		dif = 0

		
		# Si hay espacio se muestra la imagen
		if y + porcion_alto - filas_image >= 0:

			# Tomamos la sección de frame, en donde se va a ubicar
			# el gorro/tiara
			n_frame = frame[y + porcion_alto - filas_image : y + porcion_alto,
				x : x + col_image]
		else:
			# Determinamos la sección de la imagen que excede a la del video
			dif = abs(y + porcion_alto - filas_image) 
			# Tomamos la sección de frame, en donde se va a ubicar
			# el gorro/tiara
			n_frame = frame[0 : y + porcion_alto,
				x : x + col_image]

		
		# invertimos la mascara de la imagen
		mask = resized_image[:, :, 3]
		mask_inv = cv2.bitwise_not(mask)
			
		
		# creamos una imagen con fondo negro y una con la silueta en negro
		bg_black = cv2.bitwise_and(resized_image, resized_image, mask=mask)
		bg_black = bg_black[dif:, :, 0:3]
		bg_frame = cv2.bitwise_and(n_frame, n_frame, mask=mask_inv[dif:,:])

		# Sumamos las dos imágenes
		result = cv2.add(bg_black, bg_frame)
		if y + porcion_alto - filas_image >= 0:
			frame[y + porcion_alto - filas_image : y + porcion_alto, x : x + col_image] = result

		else:
			frame[0 : y + porcion_alto, x : x + col_image] = result
		
	cv2.imshow('frame',frame)

	k = cv2.waitKey(1) & 0xFF
	if k == 27:
		break
video.release()
cv2.destroyAllWindows()