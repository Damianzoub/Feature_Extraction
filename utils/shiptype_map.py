shiptype_dict = {
    range(30,40): "fishing",
    range(60,70): "passenger",
    range(70,80): "cargo",
    range(80,90): "tanker",
    range(50,60): "tug",
    range(90,100): "other",
}


def classify_shiptype(code):
    try:
        code = int(code)
    except:
        return "Unknown"
    
    for code_range,label in shiptype_dict.items():
        if code in code_range:
            return label
    
    return "unknown"

def map_shiptype(series):
    return series.map(classify_shiptype)
