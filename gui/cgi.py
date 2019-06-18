#!/usr/bin/python

import cgi, cgitb
cgitb.enable()
import logging

def print_dataframe(df):
    if df.empty:
        print("<p>empty DataFrame</p>")
    else:
        print("<table style='text-align:right;padding:20px'")
        print("<tr> <th> </th>")
        for c in df.columns:
            print("<th>{}</th>".format(c))
        print("</tr>")
        m, n = df.shape
        for i in range(m):
            if i%2:
                print("<tr><td>{}</td>".format(df.index[i]))
            else:
                print("<tr bgcolor='#DDDDDD'><td>{}</td>".format(df.index[i]))
            for j in range(n):
                print("<td>{}</td>".format(df.iat[i,j]))
            print("</tr>")
        print("</table>")


def print_row(df=None):
    """ print one row with input series [Date, Amount, Type, Description, Memo, Card, confidence] """
    if df is None:
        print("""<tr>
            <th>Date</th> <th>Amount</th> <th>Type</th> <th>Description</th> <th>Memo</th> <th>Card</th>
        </tr>""")
        return
    i = df.name
    print("<tr bgcolor='#DDDDDD'>")
    print("<td><input type='date' name='{}Date' value='{}' style='width:125px' readonly></td>".format(i,df['Date'].date()))
    print("<td><input type='number' name='{}Amount' value='{}' step='0.01' style='width:50px'></td>".format(i,df['Amount']))
    print("<td><select name='{}Type' style='width:100px'>".format(i))
    for k in np.sort(list(type_names)):
        if k==df['Type']:
            print("<option value='{}' selected>{}</option>".format(k, type_names[k]))
        else:
            print("<option value='{}'>{}</option>".format(k, type_names[k]))
    print("</select></td>")
    print("<td><input type='text' name='{}Description' value='{}' style='width:120px'></td>".format(i,df['Description']))
    print("<td><input type='text' name='{}Memo' value='{}'style='width:100px'></td>".format(i,df['Memo']))
    print("<td><select name='{}Card'>".format(i))
    for k in np.sort(card_names):
        if k==df['Card']:
            print("<option value='{0}' selected>{0}</option>".format(k))
        else:
            print("<option value='{0}'>{0}</option>".format(k))
    print("</select></td>")
    print("<td><input type='checkbox' name='{}include' value='T'></td>".format(i))
    print("</tr>")


def get_dataframe_form(form, add_transfer=False):
    index = np.unique([re.sub("[^0-9]", "", x) for x in form.keys()])
    indices = [int(x) for x in index if x!='']
    df = pd.DataFrame(columns=['Date', 'Amount', 'Type', 'Description', 'Memo', 'Card', 'include'])
    for i in indices:
        for c in df.columns:
            df.at[i,c] = form.getvalue("{}{}".format(i,c))
    df['Date'] = pd.to_datetime(df.Date)
    df['Amount'] = np.round(df.Amount.astype(float), 2)
    if add_transfer:
        select = (df.Type=='transfer') & \
                 ((df.Description=='transfer') | (df.Description=='credit card payment'))
        df_t = df[select]
        df_t['Amount'] = -df_t.Amount
        df_t['Card'] = df[select].Memo
        df_t['Memo'] = df[select].Card
        df_t.index = -df_t.index
        df = df.append(df_t)
    return df

if __name__=="__main__":
    import argparse
    import cgi
    from inspect import getmembers, isfunction

    parser = argparse.ArgumentParser(description="Useful functions for generating python cgi interface")
    parser.add_argument('--list', type=bool, nargs='?', const=1, default=0, \
                        help='list all available modules')
    parser.add_argument('--doc', type=str, \
                        help='print documents of given function')
    parser.add_argument('--log', type=str, \
                        help='indicate what level of log information to present')
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.list:
        for x in getmembers(cgi, isfunction):
            print(x[0])
        exit()
    if not FLAGS.doc is None:
        print(getattr(cgi, FLAGS.doc).__doc__)
    if not FLAGS.log is None:
        logging.basicConfig(format="%(asctime)s  %(levelname)s  %(name)s : %(message)s",
                            level=getattr(logging, FLAGS.log.upper()))
