import numpy as np
import matplotlib.pyplot as plt

def polynomial_regression(x, y, degree):
    matrixA = []
    for i in range(degree+1):
        row = []
        t = degree * 2 - i
        for j in range(degree+1):
            sxt = sum([xi**t for xi in x])
            row.append(sxt)
            t -= 1 
        matrixA.append(row)

    matrixB = []
    for i in range(degree+1):
        sxty = 0
        t = degree - i
        for j in range(len(x)):
            sxty += x[j]**t * y[j]  
        matrixB.append(sxty)   

    A = np.array(matrixA)
    B = np.array(matrixB)
    
    solution = np.linalg.solve(A, B)
    return solution

def get_polynomial_regression():
    x_input = input("Enter the values for x (comma-separated): ")
    y_input = input("Enter the values for y (comma-separated): ")
    d = int(input("Enter the degree : "))

    x = list(map(float, x_input.split(',')))
    y = list(map(float, y_input.split(',')))

    if len(x) != len(y):
        print("Error: The number of x and y values must be the same.")
        return

    solution = polynomial_regression(x, y, d)

    x_vals = np.linspace(min(x), max(x), 100)
    y_vals = sum([solution[i] * x_vals**(d - i) for i in range(d+1)])

    plt.figure(figsize=(8,6))
    plt.scatter(x, y, color='red', label='Data points')
    plt.plot(x_vals, y_vals, label=f' Regression (degree {d})', color='blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f' Regression (degree {d})')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"solution: {solution}")

get_polynomial_regression()
