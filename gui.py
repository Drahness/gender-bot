import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import webbrowser
from calculate import detect


def callback(url):
    webbrowser.open_new(url)
    
root = tk.Tk(); root.geometry("400x400"); root.title("gender-bot")

photo = tk.PhotoImage(file="image.png")

images = []

def UploadAction(event=None):
    for i in range(len(images)):
        canvas.delete(images[i])
    filename = filedialog.askopenfilename()
    img = Image.open(filename)
    img = img.resize((400, 350))
    img.save("image.png")
    canvas.photo = tk.PhotoImage(file="image.png")
    tkimg = canvas.photo
    image = canvas.create_image(0, 0, image=tkimg, anchor='nw')
    images.append(image)

def calc():
    n = detect()
    from tkinter import messagebox
    messagebox.showinfo("gender-bot", "{}".format(n))



upframe = tk.Frame(root); upframe.pack(side=tk.TOP)
op = tk.Button(upframe, text='Open', command=UploadAction, width=10); op.pack(side=tk.LEFT)

calculateb = tk.Button(upframe, text="Calculate", command=calc, width=200); calculateb.pack(side=tk.LEFT)

canvas = tk.Canvas(root, width=400, height=350, bg='white'); canvas.pack()
canvas.create_text(200, 175, text="No image selected\nsupports: BMP,EPS,GIF,JPEG,MSP,PCX,PNG,TIFF,WebP,XBM", font='System, 10')
canvas.photo = photo

link = tk.Label(root, text="https://github.com/erpk3", fg="blue", cursor="hand2", font='System 7 underline')
link.place(x=235, y=383)
link.bind("<Button-1>", lambda e: callback("https://github.com/erpk3"))

root.mainloop()
