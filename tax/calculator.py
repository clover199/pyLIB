import sys
import os
path = '\\'.join(os.path.abspath(__file__).split('\\')[:-2])
if not path in sys.path:
    sys.path.append(r'C:\Users\Emma\Documents\code\PyLib')
import re
import logging

dirs = os.path.dirname(os.path.abspath(__file__))
available = set([x.split('.')[0] for x in os.listdir(dirs) if re.match("\d\d\d\d.*\..*", x)])
years = [x[:4] for x in available]
states = [x[4:] for x in available]

from gui.tkinter_widget import *
from tax.functions import calculate_tax, taxable_from_agi, tax_rate, tax_value

class tax_gui(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.minsize(1000,600)
        self.year = None
        self.state = None
        self.status = None
        self.current_frame = tk.Frame(self.master)
        tk.Label(self.current_frame, text="This is a small app for tax calculation").grid(row=0,column=0)
        tk.Label(self.current_frame, text="--- by Ye").grid(row=1,column=1)
        self.current_frame.place(relx=.5, rely=.3, anchor="center")

        menu = tk.Menu(self, tearoff=0)
        methods = tk.Menu(self, tearoff=0)
        methods.add_command(label ='Simple', command=self.simple)
        methods.add_command(label ='Advanced', command=self.advanced)
        menu.add_cascade(label="Calculate Tax", menu=methods)
        plots = tk.Menu(self, tearoff=0)
        plots.add_command(label ='Tax value', command=self.plot_tax_value)
        plots.add_command(label ='Tax rate', command=self.plot_tax_rate)
        menu.add_cascade(label="Plot", menu=plots)
        master['menu'] = menu

    def reset_year_choices(self):
        if self.state.get()!="State":
            choices = [x for x,y in zip(years, states) if y==self.state.get()]
            choices = sorted(choices)[::-1]
            if not (self.year.get() in choices):
                self.year.var.set('Year')
            self.year.obj['menu'].delete(0, 'end')
            for choice in choices:
                self.year.obj['menu'].add_command(label=choice, command=tk._setit(self.year.var, choice))

    def reset_state_choices(self):
        if self.year.get()!='Year':
            choices = [x for x,y in zip(states,years) if y==self.year.get()]
            try:
                choices.remove('')
            except:
                pass
            choices = sorted(choices)
            if not (self.state.get() in choices):
                self.state.var.set('State')
            self.state.obj['menu'].delete(0, 'end')
            for choice in choices:
                self.state.obj['menu'].add_command(label=choice, command=tk._setit(self.state.var, choice))

    def top_frame(self, frame, func):
        top = tk.Frame(frame)
        def year_func():
            func()
            self.reset_state_choices()
        self.year = drop_down_list(top, choices=sorted(list(set(years)))[::-1], default='Year', func=year_func)
        self.year.config(width=6)
        self.year.grid(row=0, column=0, padx=2, pady=2)
        def state_func():
            func()
            self.reset_year_choices()
        self.state = drop_down_list(top, choices=sorted(list(set(states)-{''})), default='State', func=state_func)
        self.state.config(width=5)
        self.state.grid(row=0, column=1, padx=2, pady=2)
        self.status = drop_down_list(top, default='Status', func=func,
                                     choices=['Single', 'Married filing jointly',
                                              'Married filing separately', 'Head of a house hold'])
        self.status.config(width=22)
        self.status.grid(row=0, column=2, padx=2, pady=2)
        top.pack(padx=10, pady=10)

    def calc_tax(self, state, ti):
        emp = ""
        msg = "Please put a valid number"
        err = "Cannot calculate"
        year = self.year.get()
        if year=='Year': return emp
        year = int(year)
        status = self.status.get()
        if status=='Status': return emp
        if self.agi.get() in [emp,msg,err]:
            if not(ti.get() in [emp,msg,err]):
                try:
                    ti = float(ti.get())
                except Exception as error_message:
                    print(error_message)
                    ti.set(msg)
                else:
                    try:
                        return "{:,}".format(np.round(calculate_tax(year, status, y=ti, state=state),2))
                    except Exception as error_message:
                        print(error_message)
                        return err
        else:
            try:
                agi = float(self.agi.get())
            except Exception as error_message:
                print(error_message)
                self.agi.set(msg)
            else:
                if ti.get() in ["",msg,err]:
                    try:
                        y = taxable_from_agi(x=agi, year=year, state=state, status=status)
                        ti.set("{:.2f}".format(y))
                    except Exception as error_message:
                        print(error_message)
                        ti.set(err)
                else:
                    try:
                        y = float(ti.get())
                    except Exception as error_message:
                        print(error_message)
                        ti.set(msg)
                try:
                    return "{:,}".format(np.round(calculate_tax(year, status, x=agi, y=y, state=state),2))
                except Exception as error_message:
                    print(error_message)
                    return err
        return emp

    def simple_calc_tax(self):
        tax = self.calc_tax("", self.ti[0])
        self.tax[0].set(tax)
        if self.state.get()!="State":
            tax = self.calc_tax(self.state.get(), self.ti[1])
            self.tax[1].set(tax)

    def reset(self):
        for ti in self.ti:
            ti.set("")
        for tax in self.tax:
            tax.set("")

    def simple(self):
        if not (self.current_frame is None):
            self.current_frame.place_forget()
        self.agi = tk.StringVar()
        self.ti = [tk.StringVar(),tk.StringVar()]
        self.tax = [tk.StringVar(),tk.StringVar()]
        frame = tk.Frame(self.master)
        self.top_frame(frame, self.simple_calc_tax)

        mid = tk.Frame(frame)
        tk.Label(mid, text='Adjusted Gross Income\n(AGI)').grid(row=0, column=0, padx=2, pady=2)
        tk.Entry(mid, textvariable=self.agi).grid(row=0, column=1, padx=2, pady=2)
        tk.Button(mid, text='Calculate', command=self.simple_calc_tax).grid(row=0, column=2, padx=2, pady=2)
        mid.pack(padx=10)

        bot = tk.LabelFrame(frame, text='Tax value')
        tk.Label(bot, text='Taxable Income').grid(row=0, column=1, padx=2, pady=2)
        tk.Label(bot, text='Tax').grid(row=0, column=2, padx=2, pady=2)
        tk.Label(bot, text='Federal').grid(row=1, column=0, padx=2, pady=2)
        tk.Entry(bot, textvariable=self.ti[0]).grid(row=1,column=1, padx=2, pady=2)
        tk.Label(bot, textvariable=self.tax[0], width=10).grid(row=1,column=2, padx=2, pady=2)
        tk.Label(bot, text='State').grid(row=2, column=0, padx=2, pady=2)
        tk.Entry(bot, textvariable=self.ti[1]).grid(row=2,column=1, padx=2, pady=2)
        tk.Label(bot, textvariable=self.tax[1], width=10).grid(row=2,column=2, padx=2, pady=2)
        tk.Button(bot, text='Clear', command=self.reset).grid(row=2, column=3, padx=2, pady=2)
        bot.pack(padx=10, pady=10)

        frame.place(relx=.5, rely=.4, anchor="center")
        self.current_frame = frame

    def advanced(self):
        if not (self.current_frame is None):
            self.current_frame.place_forget()
        self.agi = tk.StringVar()
        self.ti = [tk.StringVar(),tk.StringVar()]
        self.tax = [tk.StringVar(),tk.StringVar()]
        frame = tk.Frame(self.master)
        self.top_frame(frame, self.simple_calc_tax)

        mid = tk.Frame(frame)
        tk.Label(mid, text='Adjusted Gross Income\n(AGI)').grid(row=0, column=0, padx=2, pady=2)
        tk.Entry(mid, textvariable=self.agi).grid(row=0, column=1, padx=2, pady=2)
        tk.Button(mid, text='Calculate', command=self.simple_calc_tax).grid(row=0, column=2, padx=2, pady=2)
        mid.pack(padx=10)

        bot = tk.LabelFrame(frame, text='Tax value')
        tk.Label(bot, text='Taxable Income').grid(row=0, column=1, padx=2, pady=2)
        tk.Label(bot, text='Tax').grid(row=0, column=2, padx=2, pady=2)
        tk.Label(bot, text='Federal').grid(row=1, column=0, padx=2, pady=2)
        tk.Button(bot, text='Clear', command=self.reset).grid(row=0, column=3, padx=2, pady=2)
        tk.Entry(bot, textvariable=self.ti[0]).grid(row=1,column=1, padx=2, pady=2)
        tk.Label(bot, textvariable=self.tax[0], width=10).grid(row=1, column=2, padx=2, pady=2)
        tk.Label(bot, text='State').grid(row=2, column=0, padx=2, pady=2)
        tk.Entry(bot, textvariable=self.ti[1]).grid(row=2,column=1, padx=2, pady=2)
        tk.Label(bot, textvariable=self.tax[1], width=10).grid(row=2, column=2, padx=2, pady=2)
        tk.Label(bot, text='____________').grid(row=3, column=0, padx=2, pady=2)
        tk.Label(bot, text='Other States').grid(row=4, column=0, padx=2, pady=2)
        add = drop_down_list(bot, choices=['IL','NJ','NY'], default="+")
        add.config(width=3)
        add.grid(row=4, column=1, padx=2, pady=2)
        bot.pack(padx=10, pady=10)

        frame.place(relx=.5, rely=.4, anchor="center")
        self.current_frame = frame

    def make_plot(self, lines, right, func, string):
        if self.year.get()=='Year':
            return
        year = int(self.year.get())
        if self.state.get()=='State':
            return
        state = self.state.get()
        if state=='Federal':
            state = ""
        if self.status.get()=='Status':
            return
        status = self.status.get()
        try:
            x, y = func(year, status, state)
        except:
            pass
        l = lines.line_interactive(x, y,
            func=lambda x,y:"\n{} {}\n{}\n".format(year, self.state.get(), status)+
            "-"*30+"\n({:.3g}, {:.3g})\n".format(x, y) + string(x,y))
        t = tk.Label(right, text="{} : {} : {}".format(year,self.state.get(),status),
                     foreground=lines.canvas.itemcget(l, 'fill'))
        def enter(enter):
            w = lines.canvas.itemcget(l, 'width')
            lines.canvas.itemconfig(l, width=float(w)+2)
            lines.canvas.tag_raise(l)
        def leave(enter):
            try:
                w = lines.canvas.itemcget(l, 'width')
                lines.canvas.itemconfig(l, width=float(w)-2)
                lines.canvas.tag_raise(l)
            except:
                pass
        def delete(enter):
            t.pack_forget()
            lines.canvas.delete(l)
        t.bind('<Enter>', enter)
        t.bind('<Leave>', leave)
        t.bind('<Button-3>', delete)
        t.pack()

    def change_plot_limit(self, frame, lines):
        lower_right = tk.Frame(frame)
        tk.Label(lower_right, textvariable=lines.text_for_interactive).pack(padx=2, pady=2)
        lim = tk.Frame(lower_right)
        tk.Label(lim, text="x from").grid(row=0, column=0, padx=2, pady=2)
        tk.Entry(lim, width=10).grid(row=0, column=1, padx=2, pady=2)
        tk.Label(lim, text="to").grid(row=0, column=2, padx=2, pady=2)
        tk.Entry(lim, width=10).grid(row=0, column=3, padx=2, pady=2)
        tk.Label(lim, text="y from").grid(row=1, column=0, padx=2, pady=2)
        tk.Entry(lim, width=10).grid(row=1, column=1, padx=2, pady=2)
        tk.Label(lim, text="to").grid(row=1, column=2, padx=2, pady=2)
        tk.Entry(lim, width=10).grid(row=1, column=3, padx=2, pady=2)
        tk.Button(lim, text='Update').grid(row=2, column=3, padx=2, pady=2)
        lim.pack(padx=2, pady=2)
        lower_right.grid(row=1, column=1, padx=10, pady=10)

    def plot_tax_value(self):
        if not (self.current_frame is None):
            self.current_frame.place_forget()
        frame = tk.Frame(self.master)

        top = tk.Frame(frame)
        self.year = drop_down_list(top, choices=sorted(list(set(years)))[::-1], default='Year')
        self.year.config(width=6)
        self.year.grid(row=0, column=0, padx=2, pady=2)
        self.state = drop_down_list(top, choices=sorted(list((set(states)|{"Federal"})-{''})), default='State')
        self.state.config(width=5)
        self.state.grid(row=0, column=1, padx=2, pady=2)
        self.status = drop_down_list(top, default='Status',
                                     choices=['Single', 'Married filing jointly',
                                              'Married filing separately', 'Head of a house hold'])
        self.status.config(width=22)
        self.status.grid(row=0, column=2, padx=2, pady=2)
        top.grid(row=0, column=0, padx=2, pady=2)

        right = tk.Frame(frame)
        tk.Label(right, text="{} : {} : {}".format("Year","State","Status")).pack()
        right.grid(row=0, column=1, padx=2, pady=2)

        canvas = tk.Frame(frame)
        lines = plot(canvas, xlim=[0,5e5], xlabel='Taxable income', grid=True)
        lines.text_for_interactive.set("\n\n\n"+"-"*30+"\n\n\n\n\n\n")
        canvas.grid(row=1, column=0, padx=10, pady=10)

        self.change_plot_limit(frame, lines)
        tk.Button(top, text='Add',
                  command=lambda:self.make_plot(lines, right, tax_value,
                  lambda x,y:"\n\n\n\n")).grid(row=0, column=3, padx=[20,2], pady=2)

        frame.place(relx=.5, rely=.5, anchor="center")
        self.current_frame = frame

    def plot_tax_rate(self):
        if not (self.current_frame is None):
            self.current_frame.place_forget()
        frame = tk.Frame(self.master)

        top = tk.Frame(frame)
        self.year = drop_down_list(top, choices=sorted(list(set(years)))[::-1], default='Year')
        self.year.config(width=6)
        self.year.grid(row=0, column=0, padx=2, pady=2)
        self.state = drop_down_list(top, choices=sorted(list((set(states)|{"Federal"})-{''})), default='State')
        self.state.config(width=5)
        self.state.grid(row=0, column=1, padx=2, pady=2)
        self.status = drop_down_list(top, default='Status',
                                     choices=['Single', 'Married filing jointly',
                                              'Married filing separately', 'Head of a house hold'])
        self.status.config(width=22)
        self.status.grid(row=0, column=2, padx=2, pady=2)
        top.grid(row=0, column=0, padx=2, pady=2)

        right = tk.Frame(frame)
        tk.Label(right, text="{} : {} : {}".format("Year","State","Status")).pack()
        right.grid(row=0, column=1, padx=2, pady=2)

        canvas = tk.Frame(frame)
        lines = plot(canvas, xlim=[0,5e5], xlabel='Taxable income', grid=True)
        lines.text_for_interactive.set("\n\n\n"+"-"*30+"\n\n\n\n\n\n")
        canvas.grid(row=1, column=0, padx=10, pady=10)

        self.change_plot_limit(frame, lines)
        tk.Button(top, text='Add',
                  command=lambda:self.make_plot(lines, right, tax_rate,
                  lambda x,y:"\n\n\n\n")).grid(row=0, column=3, padx=[20,2], pady=2)

        frame.place(relx=.5, rely=.5, anchor="center")
        self.current_frame = frame

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Basic math and statistics functions")
    parser.add_argument('--log', type=str, \
                        help='indicate what level of log information to present')
    FLAGS, unparsed = parser.parse_known_args()
    if not FLAGS.log is None:
        logging.basicConfig(format="%(asctime)s  %(levelname)s  %(name)s : %(message)s",
                            level=getattr(logging, FLAGS.log.upper()))

    tax_gui(tk.Tk()).mainloop()
