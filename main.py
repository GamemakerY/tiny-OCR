import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models
import time



img_path = input("Input the relative file path (Ex. images/test1.jpeg)")
if img_path == '':
    img_path = 'images/test1.jpeg'

start_time = time.perf_counter()

#To reduce time complexity, should help in real world scenarios, I think, 
def reduce_check(img):
    h, w, c = img.shape
    if w > 1000:
        new_w = 1000
        ar = w/h
        new_h = int(new_w/ar)
        img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_AREA)
    return img

def thresholding(image, method='gaussian'):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #No need twice, slight gaussian blur first???

    #Otsu's works better than Adaptive in some of my examples!(?)

    blur = cv2.GaussianBlur(img_gray,(5,5),0)
    if method=='otsu':
        ret3,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    elif method=='gaussian':
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    return thresh


#dilation (For lines)
#kernel = np.ones((3,15), np.uint8)
#dilated = cv2.dilate(thresh_img, kernel, iterations = 1)
def dilate(img):
    #cv2.MORPH_RECT for dilation, cv2.MORPH_Close for dilation and then erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
     (1,1)) #Why even?
    dilated = cv2.dilate(img, kernel, iterations = 1)
    return dilated


def get_characters_v2(img, img_color):
    # 1. WORD SEGMENTATION (Horizontal Dilation)
    # This merges characters into "word blobs"
    kernel = np.ones((3, 15), np.uint8) 
    dilated_words = cv2.dilate(img, kernel, iterations=1)
    
    # Find word contours
    word_contours, _ = cv2.findContours(dilated_words, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort words by Y then X
    word_rects = [cv2.boundingRect(c) for c in word_contours if cv2.boundingRect(c)[2] > 5]
    if not word_rects: return [], img_color
    
    avg_h = sum([r[3] for r in word_rects]) / len(word_rects)
    # Sort top-to-bottom, then left-to-right
    word_rects.sort(key=lambda r: (r[1] // (avg_h * 0.7), r[0]))
    
    processed_data = []
    
    for wx, wy, ww, wh in word_rects:
        # 2. CHARACTER SEGMENTATION (Inside each word)
        word_roi = img[wy:wy+wh, wx:wx+ww]
        char_contours, _ = cv2.findContours(word_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort characters inside the word left-to-right
        char_rects = [cv2.boundingRect(c) for c in char_contours if cv2.boundingRect(c)[2] > 2]
        char_rects.sort(key=lambda r: r[0])
        
        for i, (cx, cy, cw, ch) in enumerate(char_rects):
            # Convert back to global coordinates
            gx, gy = wx + cx, wy + cy
            
            # Normalization/Padding logic
            sz = int(1.5 * max(cw, ch))
            char_crop = img[gy:gy+ch, gx:gx+cw]
            
            pad_y, pad_x = (sz - ch) // 2, (sz - cw) // 2
            padded = cv2.copyMakeBorder(char_crop, pad_y, sz-ch-pad_y, pad_x, sz-cw-pad_x, 
                                        cv2.BORDER_CONSTANT, value=(0,0,0))
            
            final_img = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)
            
            processed_data.append({
                'image': final_img, 
                'x': gx, 'y': gy, 'w': cw, 'h': ch,
                'is_space_after': (i == len(char_rects) - 1) # Last char in word
            })
            
    return processed_data, img_color

'''
#plt.imshow(dilated, cmap='gray');
def get_characters(img, img_color):
    #MMAGGGICC I GOT FROM GEMINIIII 

    # 1. Find contours
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 2. Get bounding boxes and FILTER tiny noise
    rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 2 and h > 2: # Keep even small punctuation for now
            rects.append([x, y, w, h])

    if not rects: return []

    # --- ROBUST SORTING LOGIC ---
    # Sort by Y first
    rects.sort(key=lambda x: x[1])
    
    # Group into lines
    lines = []
    if rects:
        curr_line = [rects[0]]
        # Calculate a dynamic line threshold based on avg height
        avg_h = sum([r[3] for r in rects]) / len(rects)
        line_gap_threshold = avg_h * 0.5 
        
        for i in range(1, len(rects)):
            # If this char is close vertically to the previous one, same line
            if rects[i][1] < curr_line[-1][1] + line_gap_threshold:
                curr_line.append(rects[i])
            else:
                # New line detected! Sort the finished line by X
                curr_line.sort(key=lambda x: x[0])
                lines.extend(curr_line)
                curr_line = [rects[i]]
        # Sort and add the very last line
        curr_line.sort(key=lambda x: x[0])
        lines.extend(curr_line)
    
    processed_data = []
    
    # 3. Apply Dilation to the WHOLE image once to make fonts "bolder"
    # EMNIST is very thick/bold. Thin fonts fail.
    kernel = np.ones((2,2), np.uint8) # Try (3,3) if (2,2) is still too thin
    #img_bold = cv2.dilate(img, kernel, iterations=1)
    img_bold = img

    for x, y, w, h in lines:
        # PADDING: EMNIST characters are centered with a margin
        if h>4 or w>4:
            #print(f'y: {}')
            sz = int(1.5 * max(w, h)) #Original was 1.2
            char_img = img_bold[y:y+h, x:x+w]
        
            pad_y = (sz - h) // 2
            pad_x = (sz - w) // 2
        
            padded = cv2.copyMakeBorder(char_img, pad_y, sz-h-pad_y, pad_x, sz-w-pad_x, 
                                    cv2.BORDER_CONSTANT, value=(0,0,0))
        

        # FINAL STEP: Resize to 28x28
            final_img = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)
        
            processed_data.append({'image': final_img, 
            'x': x, 
            'w': w,
            'y': y,
            'h': h})
        
    return processed_data, img_color

'''
'''
def get_characters(img, img_color):
    im2 = img_color.copy()#This needs to be the original image, I was taking grayscale, for making boxes! (Make it better later)
    cropped_images=[]
    (contours, heirarchy) = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours_lines = sorted(contours, key = lambda ctr : cv2.boundingRect(ctr)[1]) #Left-to-right apparently
    for cnt in contours:
    #Only scales up right now, and also better ways to scale up!
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 1)
        square_size = int(1.2*max(w,h))
        dilated = dilate(img)
        cropped_image = dilated[y:(y+h), x:(x+w)]
        padded_image = cv2.copyMakeBorder(cropped_image, (square_size - h)//2,
                       square_size - h - ((square_size-h)//2),
                       (square_size - w)//2,
                       square_size - w - ((square_size - w)//2), cv2.BORDER_CONSTANT, value=(0,0,0))
        final_image = cv2.resize(padded_image, (28,28))
        cropped_images.append(final_image)
    return cropped_images
'''




def get_model(model_path="model.bin"):
    model = models.shufflenet_v2_x0_5()

    model.conv1[0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.fc = nn.Linear(model.fc.in_features, 47)
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = get_model()


def predict(img, model):

    img_tensor = torch.from_numpy(img).float()
    img_tensor = (img_tensor / 255.0 - 0.5) / 0.5
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad(): # Disables gradient tracking to save memory 
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        
    class_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
    return class_map[predicted.item()]



def preprocess(img_path):
    img = cv2.imread(img_path) #ex2.jpg
    img_color = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = reduce_check(img)
    img = thresholding(img, 'otsu')
    img = cv2.bitwise_not(img) #Later, have an option to check this, if the background is black already
    character_dict, img_boxes = get_characters_v2(img, img_color)
    return character_dict, img_boxes

def debug_visualize(character_list):
    n = len(character_list)
    cols = 8
    rows = (n // cols) + 1
    
    plt.figure(figsize=(15, rows * 2))
    for i, item in enumerate(character_list):
        plt.subplot(rows, cols, i + 1)
        # We show the image as the model sees it
        plt.imshow(item['image'], cmap='gray')
        # Title shows the order and the X coordinate
        plt.title(f"#{i} (x={item['x']})")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def word_segmentation(img):
    kernel = np.ones((3,15), np.uint8) #Experiment
    dilated_img = cv2.dilate(img, kernel, iterations=1)

    contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 2. Get bounding boxes and FILTER tiny noise
    rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rects.append([x, y, w, h])

    if not rects: return []

    return len(contours)





character_dict, img_boxes = preprocess(img_path)
#debug_visualize(character_dict)

pred_list=[]
img_final = img_boxes.copy()
for char in character_dict:
    prediction = predict(char['image'], model)
    cv2.putText(img_final, prediction, (char['x'], char['y']+2),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    pred_list.append(prediction)

    if char['is_space_after']:
        pred_list.append(" ")

print(*pred_list)
end_time = time.perf_counter()
elapsed_time = end_time - start_time

print(f'Total time elapsed: {elapsed_time} seconds')


plt.figure(figsize = (12,8))
plt.imshow(img_final)
plt.axis('off')
plt.show()



#https://youtu.be/9FCw1xo_s0I