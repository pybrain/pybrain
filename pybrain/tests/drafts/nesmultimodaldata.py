import pylab

ras = 'Rastrigin'
ack = 'Ackley'
wei = 'Weierstrass'
gri = 'Griewank'

functions = [ras, ack, wei, gri]
dist = [1,3,10,30, 100]

data = {1: {ras: [32, 15, 10],
            ack: [95, 91, 2],
            wei: [44, 95, 5],
            gri: [99, 100, 0], 
            },
        3: {ras: [28, 3, 21],
            ack: [95, 90, 3],
            wei: [54, 99, 7],
            gri: [96, 99, 1], 
            },
        10:{ras: [24, 16, 36],
            ack: [88, 70, 0],
            wei: [58, 101, 9],
            gri: [9,  2, 6], 
            },
        30:{ras: [20, 18, 9],
            ack: [25, 22, 21],
            wei: [58, 103, 10],
            gri: [2,  0, 17], 
            },
        100:{ras: [22, 15, 6],
            ack: [1, 4, 22],
            wei: [67, 103, 13],
            gri: [2, 0, 13], 
            },
        }



for f in functions:
    ys1 = []
    ys2 = []
    for d in dist:
        ys1.append(data[d][f][0])
        ys2.append(data[d][f][1]/float(100+data[d][f][1]))
    pylab.plot(dist, ys1, label = f+'ncg')    
    pylab.plot(dist, ys2, label = f+'cma')    
pylab.semilogx()
pylab.legend()
pylab.show()