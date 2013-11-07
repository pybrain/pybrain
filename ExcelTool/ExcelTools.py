from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.optimization.randomsearch import RandomSearch
from xlrd import open_workbook 

def readFromExcel(inCols,targetCols, numRows, fileName, offset=0, sheet=0, dataSet=None, conversionFun=None):
    """Populates a given dataset or creates a new SupervisedDataSet from an exccel file.
       
       inCols = array of colum numbers containing the input data colums, colums are indexed from 0
       targetCols = array of colum numbers containing the target data colums, colums are indexed from 0
       numRows = the number of rows ofs data
       fileName= the name of the excel file
       offset = the row the vaild data starts at
       sheet = the sheet of the workbook the data is on, indexed from 0 as it is in xlrd
       dataSet = the dataset to be populated, a SupervisedDataSet if created if it is None
       conversionFun = used to preprocess data.
    """
    book = open_workbook(fileName)
    sheet=book.sheet_by_index(sheet)
    
    if dataSet is None:
        dataSet=SupervisedDataSet(len(inCols),len(targetCols))
    for r in range(offset,(offset+numRows)):
        input=[]
        target=[]
        for inC in inCols:
            input.append(sheet.cell_value(r,inC))

        for tC in targetCols:
            target.append(sheet.cell_value(r,tC))
        try:
            if conversionFun:
                input=[conversionFun(i) for i in input]
                target=[conversionFun(t) for t in target]
                print input,target
        
            dataSet.addSample(input, target)
        except Exception:
            print 'rejected row {}'.format(r)
    return dataSet
