import os


dir = r"D:\ML\Mini Project CNN\Clg Images\1"
new_dir = r"D:\ML\Mini Project CNN\Clg Images\2"

img_list = os.listdir(dir)
print(img_list)

i = 1
for img in img_list:
    os.rename(dir + "\\" + img, new_dir + f"\\Frame {i}.jpg")
    i += 1

print(os.listdir())
