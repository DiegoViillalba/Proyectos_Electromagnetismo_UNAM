# Potential calculation, relaxation method
# Diego Villalba


import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation


# Define the relaxation method

def f(x1,x2,x3,x4):
    f = 0.25*(x1+x2+x3+x4)
    return f

# Define charge distribution
def function(i,j,n):
    # Define circle
    r = 0.05
    d = 0.003
    if (((i/n)-0.5)**2 + ((j/n)-0.5)**2) > (r-d) and (((i/n)-0.5)**2 + ((j/n)-0.5)**2) < (r+d):
        return True
    else:
        False

# Define potential function
def phi(i,j,n):
    phi = ((i/n-0.5)/0.23)**2
    return phi


# Resolution
n = 200

# Create a matrix to store the computed values
M = np.zeros((n,n))
R = np.zeros((n,n))

# Declare the initial conditions

# Remember that the matrx elemts are inverted in the display
for i in range(0,n):
    for j in range(0,n):
        if function(i,j,n)==True:
            # Potential ƒ(x,y) = sin(ø)**2
            M[i,j] = phi(i,j,n)

plt.imshow(M)
plt.colorbar()
plt.title('Condición inicial $\phi = sin^2(\theta)$')
plt.show()



# Animation elements


ims = [] # To store frames
fig, ax = plt.subplots() #Matplotlib 

im = ax.imshow(M, cmap='viridis')
ax.set_title('Evolución numérica, método relajación')
colorbar = fig.colorbar(im, ax=ax)

# Error display
error_template = 'Max error = %.4f'
err = 1
E=[[],[]]

# Loop untill error
frame=0
while err>0.01:
    
    # Reset error and storage matrix
    err = 0
    R = np.zeros((n,n))

    # Relaxation calculation
    for i in range(0,n-1):
        for j in range(0,n-1):
            if (i>0 and i < n and j>0 and j<n):
                R[i,j] = f(M[i+1,j],M[i-1,j],M[i,j+1],M[i,j-1])

    # Keeping initial conditions
    for i in range(0,n):
        for j in range(0,n):
            if function(i,j,n)==True:
                R[i,j] = phi(i,j,n)
    
    # Max error calc
    for i in range(0,n-1):
        for j in range(0,n-1):
            if abs(R[i,j]-M[i,j]) > err:
                    err = abs(M[i,j]-R[i,j])
    # Loop set
    M = R

    # Frame creation
    im = ax.imshow(M, animated=True)
    text= ax.annotate(error_template % (err),(2,5),color='white')

    if frame == 0:
        ax.imshow(M)  # show an initial one first
    
    ims.append([im,text])
    frame=frame+1

    E[0].append(frame)
    E[1].append(err)
    # To see progress
    print(err)


# Display animation
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
plt.show()

# Display iteration - error convergence

plt.plot(E[0],E[1])
plt.title('Convergencia iteración - error')
plt.ylabel('Error')
plt.xlabel('Iteración')
plt.show()

# Display final potential result
plt.imshow(M)
plt.colorbar()
plt.title('Potencial resultante $\phi = sin^2(\theta)$')
plt.show()


# Calculate the Electric field


m = 100

T = np.zeros((m,m))
U = np.zeros((m,m))

# Declare the initial conditions

# Remember that the matrx elemts are inverted in the display
for i in range(0,m):
    for j in range(0,m):
        if function(i,j,m)==True:
            # Potential ƒ(x,y) = sin(ø)**2
            T[i,j] = phi(i,j,m)

plt.imshow(T)
plt.colorbar()
plt.title('Condición inicial $\phi = sin^2(\theta)$')
plt.show()

# Error display

err = 1


# Loop untill error
frame=0
while err>0.001:
    
    # Reset error and storage matrix
    err = 0
    U = np.zeros((m,m))

    # Relaxation calculation
    for i in range(0,m-1):
        for j in range(0,m-1):
            if (i>0 and i < m and j>0 and j<m):
                U[i,j] = f(T[i+1,j],T[i-1,j],T[i,j+1],T[i,j-1])

    # Keeping initial conditions
    for i in range(0,m):
        for j in range(0,m):
            if function(i,j,m)==True:
                U[i,j] = phi(i,j,m)
    
    # Max error calc
    for i in range(0,m-1):
        for j in range(0,m-1):
            if abs(U[i,j]-T[i,j]) > err:
                    err = abs(U[i,j]-T[i,j])
    # Loop set
    T = U

plt.imshow(T)
plt.colorbar()
plt.title('Condición inicial $\phi = sin^2(\theta)$')
plt.show()

#Display electric field

gx,gy = np.gradient(T,edge_order=2)

x = np.arange(0,n,1)
y = np.arange(0,n,1)

gy = np.flipud(gy)

plt.figure()
plt.imshow(T)
plt.colorbar()
plt.quiver(-1*gy,gx,scale=5,color='green')

plt.title('Campo eléctrico')

plt.show()


# Display equipotentials

plt.figure()
plt.imshow(T,cmap='gray')
plt.contour(T)
plt.colorbar(cmap='magma')
plt.title('Equi potenciales')
plt.show()