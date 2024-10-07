import numpy as np
import math 
import tqdm
import matplotlib.pyplot as plt # side-stepping mpl backend
import matplotlib.animation as animation

# Define the relaxation method

def f(x1,x2,x3,x4):
    f = 0.25*(x1+x2+x3+x4)
    return f

# def function(i,j,n):
#     # Define circle
#     r = 0.05
#     d = 0.003
#     w = 0.3
#     p = 0.45
#     if (((i/n)-p) + w>  (r-d) and ((i/n)-p)+ w<  (r+d)) or (((i/n)-p) - w>  (r-d) and ((i/n)-p)- w<  (r+d)) or (((j/n)-p) + w>  (r-d) and ((j/n)-p)+ w<  (r+d)) or ((j/n)-p) - w>  (r-d) and ((j/n)-p)- w<  (r+d)  :
                                                                                                                                                                                

#         return True
#     else:
#         False

def function(i,j,n):
    if i>n/3 and i<(2*n/3) and j>n/3 and j<(2*n/3):
        return True
    else:
        return False

# Define potential function
def phi(i,j,n):
    phi = 100
    return phi


# Mejorando la resolución de la técnica
n = 100

# Create a matrix to store the computed values

M = np.zeros((n,n))
R = np.zeros((n,n))

# Declare the initial conditions

for i in range(0,n):
    for j in range(0,n):
        if function(i,j,n)==True:
            M[i,j] = 100

plt.imshow(M)
plt.title('Condición inicial $\phi = 100$')
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
while err>0.1:
    
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
plt.title('Potencial resultante $\phi = 100$')
plt.show()


#Display electric field

gx,gy = np.gradient(M,edge_order=2)

x = np.arange(0,n,1)
y = np.arange(0,n,1)

gy = np.flipud(gy)

plt.figure()
plt.imshow(M)
plt.colorbar()
plt.quiver(-1*gy,gx,scale=500,color='green')

plt.title('Campo eléctrico')

plt.show()


