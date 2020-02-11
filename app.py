from __future__ import absolute_import, division, print_function
import tkinter as tk
from datetime import date, datetime

from invest.useful import convert_time
import subprocess

class App(tk.Frame):

    def __init__ (self, master):
        super().__init__(master)
        self.start = tk.StringVar()
        self.end = tk.StringVar()
        self.start_value = tk.StringVar()
        self.end_value = tk.StringVar()
        tk.Label(self, text="Start:").grid(row=0, column=0, padx=2)
        tk.Entry(self, textvariable=self.start).grid(row=0, column=1, padx=2)
        tk.Label(self, text="End:").grid(row=0, column=2, padx=2)
        tk.Entry(self, textvariable=self.end).grid(row=0, column=3, padx=2)
        tk.Button(self, text="Convert", command=self.convert).grid(row=0, column=4)
        tk.Label(self, textvariable=self.start_value).grid(row=1, column=1, pady=2)
        tk.Label(self, textvariable=self.end_value).grid(row=1, column=3, pady=2)
        tk.Button(self, text="Run", command=self.run).grid(row=2, column=0)

    def convert(self):
        start = self.start.get()
        if start == "":
            start = None
        end = self.end.get()
        if end == "":
            end = None
        start, end = convert_time(start, end)
        self.start_value.set("{}".format(start))
        self.end_value.set("{}".format(end))

    def run(self):
        print("test")
        import os
        print(os.path.abspath(__file__))
        subprocess.call(r'C:\Users\Emma\Documents\PyLib\invest\analysis\etf_describe.bat')

    def show(self):
        self.pack()
        self.mainloop()

App(tk.Tk()).show()
