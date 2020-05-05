from __future__ import absolute_import, division, print_function
import tkinter as tk
import subprocess

class App(tk.Frame):

    def __init__ (self, master):
        super().__init__(master)
        tk.Label(self, text="Summary of one single stock/ETF").grid(row=0, column=0, padx=2, pady=2)
        tk.Button(self, text="Open",
            command=self.run('etf_describe.py'))\
            .grid(row=0, column=1, padx=2, pady=2)

        tk.Label(self, text="Holdings file:").grid(row=1, column=0, padx=2, pady=2)
        hold_dir = tk.StringVar()
        hold_dir.set(r"C:\Users\Emma\Documents\PyLib\invest\holdings.csv")
        tk.Entry(self, textvariable=hold_dir, width=50).grid(row=2, column=0, columnspan=2, pady=2)
        tk.Label(self, text="Quick summary of current holdings").grid(row=3, column=0, padx=2, pady=2)
        tk.Button(self, text="Open",
                  command=self.run('holdings_summary.py', hold_dir))\
            .grid(row=3, column=1, padx=2, pady=2)
        tk.Label(self, text="Performance of current holdings").grid(row=4, column=0, padx=2, pady=2)
        tk.Button(self, text="Open",
                  command=self.run('holdings_performance.py', hold_dir))\
            .grid(row=4, column=1, padx=2, pady=2)

        tk.Label(self, text="Tickers file:").grid(row=5, column=0, padx=2, pady=2)
        ticker_dir = tk.StringVar()
        ticker_dir.set(r"C:\Users\Emma\Documents\PyLib\invest\tickers.csv")
        tk.Entry(self, textvariable=ticker_dir, width=50).grid(row=6, column=0, columnspan=2, pady=2)
        tk.Label(self, text="Find allocations of portfolio").grid(row=7, column=0, padx=2, pady=2)
        tk.Button(self, text="Open",
                  command=self.run('portfolio_allocation.py', ticker_dir))\
            .grid(row=7, column=1, padx=2, pady=2)


    def run(self, file_name, para=None):
        def f():
            bat_str = r"call C:\ProgramData\Anaconda3\Scripts\activate.bat" + "\n" \
                    + r"python C:\Users\Emma\Documents\PyLib\invest\analysis" + "\\" \
                    + file_name
            if para is not None:
                bat_str += " " + para.get()
            with open(r"invest\analysis\temp.bat", "w") as f:
                f.write(bat_str)
                f.close()
            subprocess.call(r"invest\analysis\temp.bat")
        return f


    def show(self):
        self.pack()
        self.mainloop()

App(tk.Tk()).show()
