import csv
with open('P1_data.csv', 'r') as f: 
    reader = csv.reader(f)
    headers = next(reader) 
    your_list = list(reader)
h = [['0', '0', '0', '0', '0', '0','0']]
for i in your_list:
    print(i)
    if i[-1] == 'True':
        j = 0
        for x in i:  
            if x != "TRUE":
                if x != h[0][j] and h[0][j] == '0':
                    h[0][j] = x
                elif x != h[0][j] and h[0][j] != '0':
                    h[0][j] = '?'
                else:
                    pass  
            j = j + 1
print("The maximally specific hypothesis for a given training example is: ") 
print(h)
