import par_config

def getData():
    #csv = pd.read_csv("supervis.csv")
    dataX = []
    dataY = []
    with open(par_config.dataset_csv_path+par_config.dataset_csv_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            filename = row[0]
            img = Image.open(imagePath+filename)
            image = img.resize((img_width, img_height ), Image.ANTIALIAS)  #
            imge = np.array(image)
            dataX.append(imge)
            dataY.append(np.array(eval(row[1]))[0:55,0:55])
            ie = 9
        #print(len(dataX))
    return [dataX, dataY]
