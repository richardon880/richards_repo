import numpy as np
import matplotlib.pyplot as plt

# generating "unknown" data
def function(x):
    #return 2*x+3 #line
    #return 5*x**2 + 2.5 #quadratic
    #return x**4 + 5*x**3 +5*x**2 -6*x +0 #returns f(x) of quartic
    #return np.sin(x) #sine
    #return np.cos(x) #cosine
    return np.e*np.cos(x)*np.exp(-x/8) #damped oscillation
    #return 1/(x-2) # singularity at x=2
    #return np.exp(x) #exponential function
    
def dataFunc(x,noise=False,mean=0,std_dev=1):
    #data generation function
    #option to add noise to test lineSolve adaptibility to noisy data
    if noise==False: 
        return function(x) #returns function without noise
    elif noise==True:
        #adds a number from a random normal dist. with mean and std dev (automatically set)
        #function(x) returns array of values so multiply by array of different random nums same length
        return function(x)+(np.random.normal(mean,std_dev,len(x))) 

def generatePolyCoeffs(x,y, maxpower = 5):
    #create a zeroed 2D list to hold the coefficients
    coeffs = np.zeros((maxpower, maxpower))
    #iterate over the required number of polynomials
    #and get the fitting coefficients from polyfit
    for i in range(1,maxpower): 
        c = np.polyfit(x,y,i)[::-1] #reverse the list order
        coeffs[i][0:len(c)] = c #write to the zero array
    return coeffs
        
def generateFittedPolynomials(x_array,coeffs):
    shape = np.shape(coeffs)
    if len(shape) == 1: #if coefficients array passed is a singular list, this passes just the list on
    #and only iterates through the list
        coeffs= np.array([coeffs])
    #create a zero array to hold the polynomial functions
    #one row for each order of fit
    #outputs a series of y-data for each supplied row of coefficients,
    #along with the original x data
    y_fitted = np.zeros((len(coeffs), len(x_array)))
    #get each row of coefficients from coeffs, with the row number
    for row,coeff_list in enumerate(coeffs):
        #get each x point and it's location in the x data list
        for loc_of_x,x in enumerate(x_array):
            #iterate over the coefficients in the row from coeffs and compute the 
            #polynomial value of y for each x value
            #write to the corresponding row, col of y_fitted
            for power,coeff in enumerate(coeff_list):
                #could probably use array algebra for this to avoid for loop
                y_fitted[row][loc_of_x]+=coeff*x**power
    return x_array, y_fitted

def getBestFit(y,y_est,coeffs):    #find x coefficients that fit data best
    corr_list = np.zeros(1) #first item in list is a point and therefore no pearson coefficient
    for i in range(1,len(y_est)): #so replace with 0, start appending to list after first item (0)
        corr_list = np.append(corr_list,np.corrcoef(y,y_est[i])[0][1]) #filling corellation list
    print(corr_list)
    best_corr = np.amax(corr_list) #finding best fit
    location = np.where(corr_list == best_corr)[0][0] #and location
    best_fit_coeffs = coeffs[location] #returns coefficients that fit best
    return best_fit_coeffs,best_corr

def fitTest(best_corr,tol = 0.9): #checks if returned correlation coeffs are over a specified tolerance
    if best_corr >= tol:
        return True
    else:
        return False

def split(x,n=5): #splits x list/array into equal parts to iterate polyfit over smaller lengths
    #checks type of x, if array then convert to list so can be split using method shown which doesnt work for arrays
    if type(x) == "numpy.ndarray":
        x = x.tolist()
    x_split = []
    split_point = round(len(x)/n)
    for i in range(0, len(x), split_point):
        x_split.append(x[i:i+split_point])
    return x_split

def lineSolve(x,y,maxpower=5,tol=0.9): #returns algebraic solution for line data
    fit_check = False #start while loop to keep splitting if data does not fit
    count = 1 #initialising count for number of splits,n 
    while fit_check == False:
        yfit_split = [] 
        coeffs_split = []
        res = []
        best_coeffs_split = []
        print(count)
        x_split = split(x,count) #creating new split lists to iterate over each fail condition
        y_split = split(y,count)
        #print(x_split)
        #print("y_split",y_split)
        for position,lst in enumerate(x_split):
            coeffs = generatePolyCoeffs(lst, y_split[position], maxpower) #find polyfit coefficients from order = 0 to n-1 (n=5 by default)
            xvals, y_fitted = generateFittedPolynomials(lst, coeffs) #generate data points from polyfit coeffs
            yfit_split.append(y_fitted) #added split estimates and coeffs to list
            coeffs_split.append(coeffs) 
            #iterate through amount of lists y has been split into
            #return best fit coefficients for each list of y_estimates based on the different polynomials 
            best_fit_coeffs,best_fit_corr = getBestFit(y_split[position],yfit_split[position],coeffs_split[position]) #tests generated points and finds the best estimate
            res.append(fitTest(best_fit_corr,tol)) #adding whether or not best fit passed check
            best_coeffs_split.append(best_fit_coeffs)
        #print(res)
        if False in res: #if the list contains a value that does not pass redo with more splits
            count = count*2
            continue;
        else: #return the list of split x values and their coefficients
            fit_check = True
        return [x_split,best_coeffs_split]

def Range_adjust(x_highres,piecewise_data): #function adjusts high res x data to fit the bins assigned by piecewise function
    #check if x_highres is in range of piecewise data (check if first and last of each are ==)
    if x_highres[0] >= piecewise_data[0][0][0] and x_highres[-1] <= piecewise_data[0][-1][-1]:
        return x_highres #if first and last are the same pass it back
    else: #else only take x range defined by the function
        x_replacement = [] #will be returned with xrange which is inside the function
        for i in range(len(x_highres)): #iterate through each point, if inside the piecewise function, add back if not, do nothing
            if x_highres[i] >= piecewise_data[0][0][0] and x_highres[i] <= piecewise_data[0][-1][-1]:
                x_replacement.append(x_highres[i])
                return x_replacement            

def Generate_ypoint(x,coeffs): #used to generate a y value from an x value and set of
    #coefficients passed from generate y_data function
    y=0
    for power,coeff in enumerate(coeffs):
        y += (coeff*(x**power))
    return y

def Generate_ydata(start,end,npoints,piecewise): #generates a list of ydata from xdata 
    x_highres = np.linspace(start,end,npoints) #defining the higher res xdata for smoother plot 
    x_highres = Range_adjust(x_highres, piecewise) #making sure the new x range is defined by the function
    y_highres=[] #empty list to return y values
    for x in x_highres: #iterate through each x point
        for loc,x_range in enumerate(piecewise[0]): #enumerate through x data "bins" in the piecewise function
            if loc<len(piecewise[0])-1: # in the singular case the enumerate reaches the last "bin" just continue
                if x >= x_range[0] and x < piecewise[0][loc+1][0]: #if x is in the current "bin" then break, if not run again
                    break #loc variable lives on from the nested for loop above
        coeffs = piecewise[1][loc] #specific coefficients that match the bin the x point is in, will be passed to generate y_point
        y = Generate_ypoint(x,coeffs) #generating the y point
        y_highres.append(y) #appending 
    return x_highres,y_highres #returning the list

def Generate_yderiv(x,coeffs): #function calculates the derivative of  a function at a certain x point
    y=0 #called by Generate_derivative function below
    for power,coeff in enumerate(coeffs):
        y += power*coeff*x**(power-1)
    return y

def Generate_derivative(x_highres,piecewise): #generates a list of x and y points for derivative of function to be plotted
    y_deriv=[]
    for x in x_highres: #iterate through x data points
        for loc,x_range in enumerate(piecewise[0]): #enumerate through x data "bins" in the piecewise function
            if loc<len(piecewise[0])-1: #in the singular case the enumerate reaches the last "bin" just continue
                if x >= x_range[0] and x < piecewise[0][loc+1][0]: #if x is in the current "bin" then break, if not run again
                    break #loc variable lievs on from the nested loop above
        coeffs = piecewise[1][loc] #specific coefficients that match the bin the x point is in, will be passed to generate derivative function
        y = Generate_yderiv(x,coeffs) #calculating the derivative
        y_deriv.append(y) #appending
    return x_highres,y_deriv #returning

def linePlot(x_plot,y_plot,x=False,y=False): #function plots the assembled lines from line assembly
    plt.plot(x_plot,y_plot)#,c="k")
    if type(x) or (y) == "list" or "numpy.ndarray":
        #if x and y data is passed it will be plotted (for comparison purposes)
        plt.scatter(x,y,marker=".",alpha=0.15)#,c="k")
    else:
        return #if no og data passed, end function

def Data_fit(x,y,maxpower=5,tolerance=0.9,start=0,end=0,npoints=10000,return_derivative=True):
    piecewise_data = lineSolve(x, y,maxpower,tolerance); #returns piecewise function that fits data
    if start == end: #if no start or end point defined then default = start and end of piecewise data range
        start = piecewise_data[0][0][0]
        end =piecewise_data[0][-1][-1]
    x_plot,y_plot = Generate_ydata(start,end,10000, piecewise_data) #generates x and y data at range specified by start and end
    linePlot(x_plot, y_plot,x=x,y=y) #plotting the generatd piecewise function and the original data
    if return_derivative == True: 
        x_plot,y_deriv = Generate_derivative(x_plot, piecewise_data) #generates derivative at range specified above
        linePlot(x_plot,y_deriv) #plotting the derivative of the piecewise function
        plt.legend(["f(x)","df/dx"])
    return

#x = np.linspace(-20,20,300) #generate xdata, abritrary
#x = np.arange(-20,20,0.05) #alternative to above line
#y = dataFunc(x,True,0,0.1) #ydata according to specified function in data_func
#y_true = dataFunc(x,False) #non noisy data to show the actual function being plotted to compare to generated fit and noisy data
#c = generatePolyCoeffs(x,y) #find polyfit coefficients from order = 0 to n-1 (n=5 by default)
#x, y_fitted = generateFittedPolynomials(x, c) #generate data points for the polynomials
#best_fit_coeffs,best_fit_corr = getBestFit(y,y_fitted,c) #sorts through coeffs list and returns best fit coeffs
#piecewise_func = lineSolve(x, y,5,0.9); #returns list of split x points and list 
#plt.plot(x,y_true,c="Red") #plotting the true non noisy function
#x_plot,y_plot = Generate_ydata(piecewise_func[0][0][0],piecewise_func[0][-1][-1],10000, piecewise_func)
#x_plot,y_deriv = Generate_derivative(x_plot, piecewise_func)
#linePlot(x_plot, y_plot,x=x,y=y) #plotting the generatd piecewise function and the original data
#linePlot(x_plot,y_deriv) #plotting the derivative of the piecewise function

x = np.linspace(-20,20,300) #generate xdata, abritrary
y = dataFunc(x,False,0,0.2) #ydata, return noise = True, mean and std deviation of noise
Data_fit(x,y,maxpower=5,tolerance=0.99999,start=0,end=0,npoints=10000,return_derivative=True)