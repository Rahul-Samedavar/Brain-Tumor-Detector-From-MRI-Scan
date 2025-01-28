from customtkinter import CTk, CTkLabel, CTkButton, CTkFrame, CTkScrollableFrame, filedialog, CTkImage
from tkinter.messagebox import showerror, showinfo

import matplotlib.pyplot as plt
from keras.models import load_model
from util import get_probs, classes
from GradCam import *

from numpy import argmax

from PIL import Image, UnidentifiedImageError

L1 = '#BBE1FA'; L1_C = '#E1FABB'
L2 = '#3282B8'; 
D1 = '#1B262C'
D2 = '#0F4C75'

DEFAULT_MODEL_PATH = r'Models\Xception\model_epoch_16.keras'

class App(CTk):
    def __init__(self, *args, **kwargs):
        CTk.__init__(self, *args, **kwargs)
        self.title('Brain Tumour Classifier')
        self.geometry('1000x600')
        self.init_layout()

        self.mainloop()
        
    def init_layout(self):
        self.image_path = None
        self.model = None
        self.model_path = None
        try:
            self.model = load_model(DEFAULT_MODEL_PATH)
            self.model_path = DEFAULT_MODEL_PATH
        except Exception as e:
            showerror("Error", 'Could not load Default model.')

        self.config(bg=D2)

        navbar = CTkFrame(self, fg_color=L2, bg_color=L2)
        navbar.pack(side='top', fill='x')
        CTkLabel(navbar, text="Brain Tumour Classifier", font=("Roboto bold", 50), justify="left"
        ).pack(side='left', padx=10, pady=5)

        CTkButton(navbar, text='Change Model',font=('Roboto', 24), command=self.change_model,
                                fg_color=L2, hover_color=L1_C, text_color=D1,
                                corner_radius=10, border_color=D1, border_width=2
        ).pack(side='right', fill='y', padx=10, pady=10)
        
        CTkButton(navbar, text='Upload Scan',font=('Roboto', 24), command=self.select_image,
                                fg_color=L2, hover_color=L1_C, text_color=D1,
                                corner_radius=10, border_color=D1, border_width=2
        ).pack(side='right', fill='y', padx=10, pady=10)

        self.body = CTkFrame(self, fg_color='white')
        self.body.pack(side='bottom', fill='both', expand=True)

        self.default_frame = CTkFrame(self.body, fg_color='white')
        self.default_frame.grid(row=0, column=0, sticky='NSEW')

        self.default_frame.columnconfigure(0, weight=1)
        self.default_frame.rowconfigure(0, weight=1)
        self.default_frame.rowconfigure(1, minsize=70)

        self.report_frame = CTkFrame(self.body, bg_color='white', fg_color='white')
        self.report_frame.grid(row=0, column=0, sticky='NSEW')

        self.image_lbl = CTkLabel(self.report_frame, text='')
        self.image_lbl.pack(side='left', fill='y', anchor='center', padx=10)
        
        rep = CTkFrame(self.report_frame, bg_color='transparent', fg_color='transparent', border_color=D2, border_width=1, corner_radius=0)
        rep.pack(side='right', fill='both', expand=True, padx=5, pady=15)


        self.rep_class = CTkLabel(rep, text="No Tumor", font=('Roboto bold', 30))
        self.rep_class.grid(row=1, column=1, columnspan=2, sticky="EW", padx=5, pady=5)
        
        for i, txt in enumerate(["Class", "Probability"]):
            CTkLabel(rep, text=txt, font=('Roboto bold', 20)
            ).grid(row=2, column=i+1, sticky="W", padx=5, pady=5)

        for i, txt in enumerate(["No Tumor", "Glioma Tumor", "Meningionma Tumor", "Pituitary Tumor"]):
            CTkLabel(rep, text=txt, font=('Roboto', 18), width=200, anchor="w"
            ).grid(row=i+3, column=1, sticky="W", padx=5, pady=5)

        n_prob = CTkLabel(rep, text="", font=('Roboto', 18), anchor="w")
        n_prob.grid(row=3, column=2, sticky="EW", padx=5, pady=5)

        g_prob = CTkLabel(rep, text="", font=('Roboto', 18), anchor="w")
        g_prob.grid(row=4, column=2, sticky="EW", padx=5, pady=5)

        m_prob = CTkLabel(rep, text="", font=('Roboto', 18), anchor="w")
        m_prob.grid(row=5, column=2, sticky="EW", padx=5, pady=5)
        
        p_prob = CTkLabel(rep, text="", font=('Roboto', 18), anchor="w")
        p_prob.grid(row=6, column=2, sticky="EW", padx=5, pady=5)

        # Arranged according to class indices of model
        self.prob_lbls = [g_prob, m_prob, n_prob, p_prob]

        gc_btn = CTkButton(rep, text='Grad Cam',font=('Roboto', 20), command=self.display_gc,
                                fg_color=L1, hover_color=L1_C, text_color=D1,
                                corner_radius=7, border_color=D1, border_width=2
        )
        gc_btn.grid(row=7, column=1, sticky='W', padx=5, pady=5)
        
        gc_btn = CTkButton(rep, text='Grad Cam++',font=('Roboto', 20), command=self.display_gcpp,
                        fg_color=L1, hover_color=L1_C, text_color=D1,
                        corner_radius=7, border_color=D1, border_width=2
        )
        gc_btn.grid(row=7, column=2, sticky='W', padx=5, pady=5)

        rep.columnconfigure([0, 3], weight=1)
        rep.rowconfigure([0,8], weight=1, minsize=1)

        self.default_frame.tkraise()

        self.body.columnconfigure(0, weight=1)
        self.body.rowconfigure(0, weight=1)

    def change_model(self):
        file_name = filedialog.askopenfilename(initialdir="./Models/", filetypes=(['.keras', 'Keras'], ['.h5', 'H5']))
        if (self.model is None or self.model_path != file_name):
            try:
                self.model = load_model(file_name)
                self.model_path = file_name
                showinfo("Success", f"Model Loaded.\npath: {file_name}")
                if (self.image_path is not None):
                    self.predict()
            except OSError as e:
                showerror("Invalid", "Choose a valid model file.")
            except Exception as e:
                showerror("Error", f"Failed to load model.\n{str(e)}")

    def select_image(self):
        file_name = filedialog.askopenfilename(initialdir="./", filetypes=(['.jpg', 'JPG'], ['.jpeg', 'JPEG'], ['.png', 'PNG']))
        if((self.image_path is None or self.image_path != file_name) and self.model is not None):
            try:
                im = Image.open(file_name)
                h, w  = im.size
                img = CTkImage(im, size=(500, (h*500/w)))
                self.image_lbl.configure(image=img)
                self.image_path = file_name
                self.report_frame.tkraise()
                self.predict()

            except UnidentifiedImageError as e:
                showerror("Invalid", "Invalid image\nchoose a valid image")

            except Exception as e:
                showerror("Error", str(e))


    def predict(self):
        try:
            probs = get_probs(self.image_path, self.model)
            for prob, lbl in zip(probs, self.prob_lbls):
                lbl.configure(text=f"{100*prob:.6f}%")
            class_ind = argmax(probs)
            self.rep_class.configure(text=classes[class_ind])

        except Exception as e:
            showerror("Error", str(e))
        

    def display_gc(self):
        if (self.model and self.image_path):
            grad_cam_visualize(self.model, self.image_path, self.get_last_layer()).show()

    def display_gcpp(self):
        if (self.model and self.image_path):
            grad_cam_plus_plus_visualize(self.model, self.image_path, self.get_last_layer()).show()

    def get_last_layer(self):
        if self.model:
            for layer in self.model.layers[::-1]:
                if 'conv' in layer.name:
                    return layer.name
        return ''
        
app = App()