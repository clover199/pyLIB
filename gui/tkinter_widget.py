import numpy as np
import pandas as pd
import tkinter as tk
import logging


# 2019-7-28
class display_dataframe(tk.Frame):
    def __init__(self, master, df, fontsize=10):
        """
        Display pandas DataFrame
        input:  master      the frame to display
                df          the pandas DataFrame
                fontsize    the size of texts, default 10
        """
        super().__init__(master)
        tk.Label(self, text='{}'.format(df.index.name), bg='white',
            font='TkTextFont {} bold'.format(fontsize)).\
            grid(row=0, column=0, padx=2, pady=2)
        for i, col in enumerate(df.columns):
            tk.Label(self, text="{}".format(col), bg='white',
                font='TkTextFont {} bold'.format(fontsize)).\
                grid(row=0, column=i+1, padx=2, pady=2)
        bg = ['#DDDDDD'] * df.shape[0]
        bg[1::2] = ['white'] * (df.shape[0] // 2)
        for i, row in enumerate(df.index):
            tk.Label(self, text="{}".format(row), bg=bg[i],
                font='TkTextFont {} bold'.format(fontsize))\
                .grid(row=i+1, column=0, padx=2, pady=2, sticky='WE')
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                tk.Label(self, text="{}".format(df.iloc[i,j]), bg=bg[i],
                    font='TkTextFont {}'.format(fontsize))\
                    .grid(row=i+1, column=j+1, padx=2, pady=2, sticky='WE')


# 2019-5-18
class entry_table(tk.Frame):
    def __init__(self, master, df, show_index=True, show_column=True, width=None, bg=None):
        """
        input with a dataframe with elements as tk.StringVar
        input:  master      the frame to put entries
                df          the pandas DataFrame with elements as tk.StringVar
                show_index  indicate whether to show index of df, default True
                show_column indicate whether to show columns of df, default True
                width       the width of each column, default None
                            can be a number or a list
                bg          the background color for each row, default None
                            can be a number of a list
        """
        super().__init__(master)
        if show_column:
            for i, col in enumerate(df.columns):
                tk.Label(self, text="{}".format(col)).grid(row=0, column=i+1, padx=2, pady=2)
        if show_index:
            for i, row in enumerate(df.index):
                tk.Label(self, text="{}".format(row)).grid(row=i+1, column=0, padx=2, pady=2)
        if width is None:
            width = [None] * df.shape[1]
        if np.array(width).ndim==0:
            width = [width] * df.shape[1]
        if bg is None:
            bg = [None] * df.shape[0]
        if np.array(bg).ndim==0:
            bg = [bg] * df.shape[0]
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                tk.Entry(self, textvariable=df.iat[i,j], width=width[j], bg=bg[i])\
                    .grid(row=i+1, column=j+1)


class scroll_list(tk.Frame):
    def __init__(self, master=None, choices=[], sort=False):
        super().__init__(master)
        self._sorted_ = sort
        scrollbar = tk.Scrollbar(self)
        scrollbar.pack(side='right', fill=tk.Y, padx=[0,5], pady=5)
        self.listbox = tk.Listbox(self, selectmode=tk.EXTENDED)
        self.listbox.pack(side='right', padx=[5,0], pady=5)
        if sort:
            choices = np.sort(choices)
        for c in choices:
            self.listbox.insert(tk.END, c)
        self.listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.listbox.yview)

    def get(self):
        return [self.listbox.get(x) for x in self.listbox.curselection()]

    def reset(self, choices):
        self.listbox.delete(0, tk.END)
        if self._sorted_:
            choices = np.sort(choices)
        for c in choices:
            self.listbox.insert(tk.END, c)

    def insert(self, choices):
        if self._sorted_:
            choices = np.append(self.listbox.get(0,tk.END), choices)
            self.reset(np.sort(choices))
        else:
            for c in choices:
                self.listbox.insert(tk.END, c)

    def delete(self, choice):
        choices = self.listbox.get(0,tk.END)
        try:
            loc = list(choices).index(choice)
            self.listbox.delete(loc)
        except:
            print(choice, "does not exist")


class left_right_list(tk.Frame):
    def __init__(self, master=None, left_choices=[], right_choices=[],
                 left_name="", right_name="", command=None):
        super().__init__(master)
        tk.Label(self, text=left_name).grid(row=0, column=0)
        self.left = scroll_list(self, choices=left_choices, sort=True)
        self.left.grid(row=1, rowspan=5, column=0)
        tk.Label(self, text=right_name).grid(row=0, column=2)
        self.right = scroll_list(self, choices=right_choices, sort=True)
        self.right.grid(row=1, rowspan=5, column=2)

        def func(orig, dest):
            move = orig.get()
            dest.insert(move)
            for c in move:
                orig.delete(c)
            if command is not None:
                command()

        tk.Button(self, text="->", command=lambda:func(self.left, self.right))\
            .grid(row=2, column=1, padx=5, pady=20)
        tk.Button(self, text="<-", command=lambda:func(self.right, self.left))\
            .grid(row=4, column=1, padx=5, pady=20)

    def get_left(self):
        return self.left.listbox.get(0,tk.END)

    def get_right(self):
        return self.right.listbox.get(0,tk.END)


class drop_down_list(tk.Frame):
    def __init__(self, master=None, choices=['Choices'], default=None, func=lambda:""):
        super().__init__(master)
        self.logger = logging.getLogger(__name__)
        self.var = tk.StringVar(self)
        if not (default is None):
            self.var.set(default)
        self.obj = tk.OptionMenu(self, self.var, *choices)
        self.obj.pack()
        self.var.trace('w', lambda *_:func())

    def get(self):
        return self.var.get()

    def set(self):
        return self.var.set()

    def config(self, **kwarg):
        self.obj.config(**kwarg)


class my_entry(tk.Frame):
    def __init__(self, master, name="", default="", command=None, button_name=""):
        super().__init__(master)
        self.text = tk.StringVar()
        self.text.set(default)
        name = tk.Label(self, text=name)
        name.pack(side=tk.LEFT, padx=10)
        self.entry = tk.Entry(self, textvariable=self.text)
        self.entry.pack(side=tk.LEFT)
        if button_name!="":
            tk.Button(self, text=button_name, command=command).pack(side=tk.LEFT, padx=10)

    def get(self):
        return self.text.get()


class my_scale(tk.Frame):
    def __init__(self, master, values, labels=None, default=None, orient=tk.HORIZONTAL,
                 command=None, name=""):
        super().__init__(master)
        self._logger_ = logging.getLogger(__name__)
        l = len(values)
        self._val_ = dict(zip(np.arange(l), values))
        self._val_rev_ = dict(zip(values, np.arange(l)))
        if labels is None:
            labels = values
        self._label_ = dict(zip(np.arange(l), labels))
        self._text_ = tk.StringVar()
        self._text_.set(labels[0])
        tk.Label(self, text=name).grid(row=0,column=0)
        tk.Label(self, textvariable=self._text_).grid(row=0,column=2)
        tk.Label(self, text="{}".format(labels[0])).grid(row=1,column=1)
        loc = tk.IntVar()
        loc.set(0)
        if default is not None:
            try:
                index = self._val_rev_[default]
                loc.set(index)
                self._text_.set(self._label_[int(index)])
            except:
                self._logger_.warning("Default value not set")
        self.scale = tk.Scale(self, from_=0, to=l-1, orient=orient, showvalue=0,
                              command=self._set_text_, variable=loc)
        self.scale.grid(row=1, column=2)
        self.command = command
        tk.Label(self, text="{}".format(labels[-1])).grid(row=1,column=3)

    def _set_text_(self, val):
        self._text_.set(self._label_[int(val)])
        if self.command is not None:
            self.command(val)

    def get(self):
        return self._val_[self.scale.get()]

    def set(self, val):
        self._logger_.debug("set value {} for my_scale".format(val))
        self.scale.set(self._val_rev_[val])

# 2019-7-4
class plot_embed(tk.Frame):
    def __init__(self, master, fig=None, x_scroll=False, y_scroll=False):
        super().__init__(master)
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        if fig is None:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            fig.add_subplot(111)
        self.fig = fig
        self.canvas = tk.Canvas(self)
        self.canvas.grid(row=0, column=0, sticky='SNEW')
        self._canvas_ = FigureCanvasTkAgg(fig, self.canvas)
        window = self._canvas_.get_tk_widget()
        window.pack()

        if x_scroll:
            hbar = tk.Scrollbar(self, orient=tk.HORIZONTAL)
            hbar.grid(row=1, column=0, sticky='EW')
            hbar.config(command=self.canvas.xview)
            self.canvas.config(xscrollcommand=hbar.set)

        if y_scroll:
            vbar = tk.Scrollbar(self)
            vbar.grid(row=0, column=1, sticky='SN')
            vbar.config(command=self.canvas.yview)
            self.canvas.config(yscrollcommand=vbar.set)

        if x_scroll or y_scroll:
            self.canvas.create_window(0, 0, window=window)
            self.canvas.config(scrollregion=self.canvas.bbox(tk.constants.ALL))


    def show(self):
        self._canvas_.show()

    def fig_config(self, **kwarg):
        self._canvas_.get_tk_widget().config(kwarg)

# 2019-7-4
class plot_embed_toolbar(tk.Frame):
    def __init__(self, master, fig=None):
        super().__init__(master)
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
        if fig is None:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            fig.add_subplot(111)
        self.fig = fig
        self._canvas_ = FigureCanvasTkAgg(fig, self)
        self._canvas_.get_tk_widget().pack()
        toolbar = NavigationToolbar2TkAgg(self._canvas_, self)
        toolbar.update()

    def show(self):
        self._canvas_.show()

    def config(self, **kwarg):
        self._canvas_.get_tk_widget().config(kwarg)


class plot(tk.Frame):
    def __init__(self, window=None, figsize=[600,400],
                 xlabel="", ylabel="", title="", axes=[0.1,0.07,0.86,0.9],
                 axes_left=0, axes_right=0, axes_bottom=0, axes_top=0,
                 xlim=None, ylim=None, grid=False):
        """
        figsize = [width, height]
        xlabel = ""
        ylabel = ""
        title = ""
        axes = [origin_x, origin_y, length_x, length_y]
        axes_left = 0 changes to left margin
        axes_right = 0 changes to right margin
        axes_bottom = 0 changes to bottom margin
        axes_top = 0 changes to top margin
        xlim = [xmin, xmax]
        ylim = [ymin, ymax]
        grid = False
        """
        super().__init__(window)
        self.logger = logging.getLogger(__name__)
        import sys
        import os
        path = '\\'.join(os.path.abspath(__file__).split('\\')[:-2])
        if not path in sys.path:
            sys.path.insert(0, path)
        self.window = window
        self.text_for_interactive = tk.StringVar()
        self._w_, self._h_ = figsize
        self.canvas = tk.Canvas(self.window, bg="white", width=self._w_, height=self._h_)
        self.canvas.grid(row=0,column=0)

        self._xlabel_ = xlabel
        if xlabel!="":
            axes_bottom += 0.05
        self._ylabel_ = ylabel
        if ylabel!="":
            axes_top += 0.05
        self._title_ = title
        if title!="":
            axes_top += 0.02
        axes = [axes[0]+axes_left, axes[1]+axes_bottom, axes[2]-axes_left-axes_right, axes[3]-axes_bottom-axes_top]
        self._l_, self._r_ = axes[0]*self._w_, (axes[0]+axes[2])*self._w_
        self._b_, self._t_ = (1-axes[1])*self._h_, (1-axes[1]-axes[3])*self._h_

        self._xlim_ = xlim
        self._ylim_ = ylim
        self._grid_ = grid

        self._xticker_size_ = 0.015
        self._yticker_size_ = 0.015
        self._axes_set_ = False
        self._x_ = None
        self._y_ = None
        self._colors_ = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
                         '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
        self._lines_ = {}

    def _set_tickers_(self, a, b):
        num = 6
        if a==b:
            a = 0
            diff = float( "{:.1g}".format( 2*b / (num-1)) )
        else:
            diff = float( "{:.1g}".format( (b-a) / (num-1)) )
            a = np.floor(a/diff) * diff
        b = np.ceil(b/diff) * diff
        tickers = np.arange(a, b+diff/2, diff)
        return tickers, ["{:g}".format(x) for x in tickers]

    def _set_lim_ticker_(self):
        from basic.mathe import pwl_value
        if self._x_ is None:
            self._x_ = [0,1]
            self._y_ = [0,1]

        if self._xlim_ is None:
            self._xticker_, self._xticker_label_ = self._set_tickers_(np.min(self._x_), np.max(self._x_))
            self._xlim_ = [self._xticker_[0], self._xticker_[-1]]
        else:
            self._xticker_, self._xticker_label_ = self._set_tickers_(np.min(self._xlim_), np.max(self._xlim_))

        if self._ylim_ is None:
            arg = (self._xlim_[0]<=self._x_) & (self._x_<=self._xlim_[1])
            ymin = np.min(self._y_[arg])
            ymax = np.max(self._y_[arg])
            if arg[0]==False:
                yleft = pwl_value(self._x_, self._y_, self._xlim_[0])
                ymin = min(ymin, yleft)
                ymax = max(ymax, yleft)
            if arg[-1]==False:
                yright = pwl_value(self._x_, self._y_, self._xlim_[1])
                ymin = min(ymin, yright)
                ymax = max(ymax, yright)
            self._yticker_, self._yticker_label_ = self._set_tickers_(ymin, ymax)
            self._ylim_ = [self._yticker_[0], self._yticker_[-1]]
        else:
            self._yticker_, self._yticker_label_ = self._set_tickers_(np.min(self._ylim_), np.max(self._ylim_))

    def _xaxis_(self):
        for ticker, label in zip(self._xticker_, self._xticker_label_):
            if self._xlim_[0]<=ticker and ticker<=self._xlim_[1]:
                loc = self._l_ + (self._r_-self._l_)*(ticker-self._xlim_[0])/(self._xlim_[1]-self._xlim_[0])
                self.canvas.create_line(loc, self._b_, loc, self._b_+self._xticker_size_*self._h_,
                                        fill='black', tags='xtickers')
                self.canvas.create_text(loc, self._b_+self._xticker_size_*self._h_*1.5,
                                        text=label, anchor=tk.N, tags='xtickerlabels')

    def _yaxis_(self):
        for ticker, label in zip(self._yticker_, self._yticker_label_):
            if self._ylim_[0]<=ticker and ticker<=self._ylim_[1]:
                loc = self._b_ + (self._t_-self._b_)*(ticker-self._ylim_[0])/(self._ylim_[1]-self._ylim_[0])
                self.canvas.create_line(self._l_-self._yticker_size_*self._h_, loc, self._l_, loc,
                                        fill='black', tags='ytickers')
                self.canvas.create_text(self._l_-self._yticker_size_*self._w_*1.5, loc,
                                        text=label, anchor=tk.E, tags='ytickerlabels')

    def _show_grid_(self):
        for ticker in self._xticker_:
            if self._xlim_[0]<ticker and ticker<self._xlim_[1]:
                loc = self._l_ + (self._r_-self._l_)*(ticker-self._xlim_[0])/(self._xlim_[1]-self._xlim_[0])
                self.canvas.create_line(loc, self._b_, loc, self._t_, fill='grey', tags='xgrid')
        for ticker in self._yticker_:
            if self._ylim_[0]<ticker and ticker<self._ylim_[1]:
                loc = self._b_ + (self._t_-self._b_)*(ticker-self._ylim_[0])/(self._ylim_[1]-self._ylim_[0])
                self.canvas.create_line(self._l_, loc, self._r_, loc, fill='grey', tags='ygrid')

    def _axes_setup_(self):
        self._axes_set_ = True
        self._set_lim_ticker_()
        self.canvas.create_line(self._l_, self._b_, self._r_, self._b_, self._r_, self._t_,
                                self._l_, self._t_, self._l_, self._b_, fill='black')
        if self._xlabel_!="":
            self.canvas.create_text((self._r_+self._l_)/2, self._b_+(self._xticker_size_)*self._h_*5,
                                    text=self._xlabel_, anchor=tk.N)
        if self._ylabel_!="":
            self.canvas.create_text(self._l_-self._yticker_size_*self._w_*5, (self._t_+self._b_)/2,
                                    text=self._ylabel_, anchor=tk.E)
        if self._title_!="":
            self.canvas.create_text((self._r_+self._l_)/2, self._t_-self._h_*0.01,
                                    text=self._title_, anchor=tk.S, font=10)
        self._xaxis_()
        self._yaxis_()

        if self._grid_:
            self._show_grid_()

    def _convert_line_(self, x, y):
        from basic.mathe import pwl_value
        x = np.array(x).flatten()
        y = np.array(y).flatten()
        assert len(x)==len(y)
        # I do not know what to do if the line exceeds ylim
        arg = (self._xlim_[0]<=x) & (x<=self._xlim_[1]) # & (self._ylim_[0]<=y) & (y<=self._ylim_[1])
        cx = x[arg]
        cy = y[arg]
        if arg[0]==False:
            cx = np.append(self._xlim_[0], cx)
            cy = np.append(pwl_value(x, y, self._xlim_[0]), cy)
        if arg[-1]==False:
            cx = np.append(cx, self._xlim_[-1])
            cy = np.append(cy, pwl_value(x, y, self._xlim_[-1]))
        cx = self._l_ + (self._r_-self._l_)*(cx-self._xlim_[0])/(self._xlim_[1]-self._xlim_[0])
        cy = self._b_ + (self._t_-self._b_)*(cy-self._ylim_[0])/(self._ylim_[1]-self._ylim_[0])
        return cx, cy

    def _convert_point_(self, x, y):
        x = np.array(x).flatten()
        y = np.array(y).flatten()
        assert len(x)==len(y)
        arg = (self._xlim_[0]<=x) & (x<=self._xlim_[1]) & (self._ylim_[0]<=y) & (y<=self._ylim_[1])
        x = self._l_ + (self._r_-self._l_)*(x-self._xlim_[0])/(self._xlim_[1]-self._xlim_[0])
        y = self._b_ + (self._t_-self._b_)*(y-self._ylim_[0])/(self._ylim_[1]-self._ylim_[0])
        return x[arg], y[arg]

    def _convert_inverse_(self, x, y):
        x = np.array(x).flatten()
        y = np.array(y).flatten()
        assert len(x)==len(y)
        x = self._xlim_[0] + (self._xlim_[1]-self._xlim_[0])*(x-self._l_)/(self._r_-self._l_)
        y = self._ylim_[0] + (self._ylim_[1]-self._ylim_[0])*(y-self._b_)/(self._t_-self._b_)
        return x, y

    def line(self, x, y, color=None, width=2):
        self._x_, self._y_ = np.array(x), np.array(y)
        if not self._axes_set_:
            self._axes_setup_()
        x, y = self._convert_line_(self._x_, self._y_)
        points = np.vstack([x,y]).T.flatten()
        if color is None:
            color = self._colors_[0]
            self._colors_ = np.roll(self._colors_, -1)
        l = self.canvas.create_line(*points, fill=color, width=width, tags='line')
        self._lines_[l] = [self._x_, self._y_]
        return l

    def scatter(self, x, y, color=None, size=3):
        self._x_, self._y_ = np.array(x), np.array(y)
        if not self._axes_set_:
            self._axes_setup_()
        x, y = self._convert_point_(self._x_, self._y_)
        points = np.vstack([x,y]).T.flatten()
        if color is None:
            color = self._colors_[0]
            self._colors_ = np.roll(self._colors_, -1)
        for i, j in zip(x,y):
            self.canvas.create_oval(i-size, j-size, i+size, j+size,
                                    fill=color, outline=color, tags='point')

    def _line_interactive_enter_(self, enter, line, xs, ys, func):
        from basic.mathe import pwl_value
        ex, ey = enter.x, enter.y
        x, y = self._convert_inverse_([ex], [ey])
        x = np.clip(x, *self._xlim_)[0]
        y = np.clip(pwl_value(xs, ys, x), *self._ylim_)
        ex, ey = self._convert_line_([x], [y])
        ex = ex[0]
        ey = ey[0]
        w = self.canvas.itemcget(line, 'width')
        self.canvas.itemconfig(line, width=float(w)+2)
        self.text_for_interactive.set(func(x,y))
        self._pt_ = self.canvas.create_oval(ex-5,ey-5,ex+5,ey+5, fill='black')
        self.canvas.tag_raise(line)

    def _line_interactive_leave_(self, enter, line):
        w = self.canvas.itemcget(line, 'width')
        self.canvas.itemconfig(line, width=float(w)-2)
        self.canvas.delete(self._pt_)

    def line_interactive(self, x, y, color=None, width=2,
            func=lambda x,y: "({:.2g}, {:.2g})".format(x, y)+"\n"+"-"*25+"\n"):
        l = self.line(x, y, color, width)
        self.canvas.tag_bind(l, "<Enter>", lambda enter:self._line_interactive_enter_(enter, l, x, y, func))
        self.canvas.tag_bind(l, "<Leave>", lambda enter:self._line_interactive_leave_(enter, l))
        return l

    def set_lim(self, xmin=None, xmax=None, ymin=None, ymax=None):
        if (self._xlim_ is None) or (self._ylim_ is None):
            return
        if (xmin is None) and (xmax is None) and (ymin is None) and (ymax is None):
            return
        if not (xmin is None):
            self._xlim_[0] = xmin
        if not (xmax is None):
            self._xlim_[1] = xmax
        self._xlim_ = [min(self._xlim_), max(self._xlim_)]
        if not (ymin is None):
            self._ylim_[0] = ymin
        if not (ymax is None):
            self._ylim_[1] = ymax
        self._ylim_ = [min(self._ylim_), max(self._ylim_)]

        for line in self.canvas.find_withtag('line'):
            x, y = self._convert_line_(*self._lines_[line])
            points = np.vstack([x,y]).T.flatten()
            self.canvas.coords(line, *points)

        self._set_lim_ticker_()
        self.canvas.delete('xtickers')
        self.canvas.delete('xtickerlabels')
        self._xaxis_()
        self.canvas.delete('ytickers')
        self.canvas.delete('ytickerlabels')
        self._yaxis_()

        if self._grid_:
            self.canvas.delete('xgrid')
            self.canvas.delete('ygrid')
            self._show_grid_()

    def show(self):
        if not self._axes_set_:
            self._axes_setup_()
        self.mainloop()


if __name__=="__main__":
    import argparse
    import tkinter_widget
    from inspect import getmembers, isfunction, isclass

    parser = argparse.ArgumentParser(description="Basic math and statistics functions")
    parser.add_argument('--list', type=bool, nargs='?', const=1, default=0, \
                        help='list all available modules')
    parser.add_argument('--doc', type=str, \
                        help='print documents of given function')
    parser.add_argument('--log', type=str, \
                        help='indicate what level of log information to present')
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.list:
        print("Functions")
        for x in getmembers(tkinter_widget, isfunction):
            print("\t", x[0])
        print("Classes")
        for x in getmembers(tkinter_widget, isclass):
            print("\t", x[0])
        exit()
    if not FLAGS.doc is None:
        print(getattr(tkinter_widget, FLAGS.doc).__doc__)
        exit()
    if not FLAGS.log is None:
        logging.basicConfig(format="%(asctime)s  %(levelname)s  %(name)s : %(message)s",
                            level=getattr(logging, FLAGS.log.upper()))
