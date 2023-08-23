import numpy as np
import matplotlib.pyplot as plt


def myfun(v, u, dy, dx):
    '''
    v  : (often) integer value
    u  : (often) integer value
    dy : frequency along vertical axis
    dx : frequency along horizontal axis


     cos(-2pi * (ux + vy)) = 1 
         when - 2 pi * (ux + vy) = 2 * pi *n  (n=0,1,..)
         i.e.,          ux + vy = - n 
        this function taks its maximum value 1
    '''
    term = -2. * np.pi * (u * dx + v * dy)
    return (np.cos(term))


def get_sinusoid_img(dy, dx, _M, _N):
    orig_img = np.zeros((_N, _M))
    startx = - _M // 2
    starty = - _N // 2
    countx = -1
    for x in range(startx, startx + _M):
        countx += 1
        county = -1
        for y in range(starty, starty + _N):
            county += 1
            orig_img[county, countx] = myfun(y, x, dy, dx)
    return (orig_img)


def get_wavelength_adjusted_vec(dx, dy):
    dx_dist = dx
    dy_dist = dy
    xynorm = np.sqrt(dx ** 2 + dy ** 2)
    if xynorm == 0:
        wavelength = 0
        _dx, _dy =0# Handle the case of zero division
    else:
        wavelength = 1 / np.sqrt(dx_dist ** 2 + dy_dist ** 2)
        _dx, _dy = dx / xynorm * wavelength, dy / xynorm * wavelength
    return (_dx, _dy, wavelength)

def F(v,u,fs): ## y = v (N), x = u (M)
    _F = 0
    N, M = fs.shape
    startu = - fs.shape[1]/2
    startv = - fs.shape[0]/2
    xcount = -1
    for x in range(int(startu), int(startu) + fs.shape[1]):
        xcount += 1
        ycount  = -1
        for y in range(int(startv), int(startv) + fs.shape[0]):
            ycount += 1
            term = x*u/float(M) + y*v/float(N)
            _F += fs[ycount,xcount]*np.exp(2.0*np.pi*1j*term)
    return(_F)
def f(y,x,Fs):
    _f = 0
    N, M = Fs.shape
    startx = - Fs.shape[1]/2
    starty = - Fs.shape[0]/2
    ucount = - 1
    for u in range(int(startx), int(startx) + Fs.shape[1]):
        ucount += 1
        vcount  = -1
        #print("u={:2.0f} count={:2.0f}".format(u,ucount))
        for v in range(int(starty), int(starty) + Fs.shape[0]):

            vcount += 1
            term = x*u/float(M) + y*v/float(N)
            _f += Fs[vcount,ucount]*np.exp(-2.0*np.pi*1j*term)
    _f /= np.prod(Fs.shape)
    return(_f)
def plot_nonzero_Fourier_coefficient_line(_y,_x,_N,_M):
    ## horizontal and vertical lines along nonzero Fourier coefficient
    if _x < _M/2: ## x is always positive
        plt.axvline(_x)
        plt.axvline(-_x)
    else: ## x is not between - M/2,..., M/2-1
        plt.axvline(_M - _x)
        plt.axvline(-_M + _x)
    if _y < _N/2:
        plt.axhline(_y)
        plt.axhline(-_y)
    else: ## y is not between - N/2,..., N/2-1
        plt.axhline(_N - _y)
        plt.axhline(-_N + _y)

def wavelength():
    _N = 18  ## the number of pixels along y axis
    _M = 42  ## the number of pixels along x axis
    _ys = [0, 3, 7, 5]
    _xs = [10, 0, 1, 4]
    for _y, _x in zip(_ys, _xs):
        dy = _y / float(_N)
        dx = _x / float(_M)
        orig_img = get_sinusoid_img(dy, dx, _M, _N)
        _dx, _dy, wavelength = get_wavelength_adjusted_vec(dx, dy)

        const = 0.5

        fig = plt.figure(figsize=(_M * const, _N * const))
        plt.imshow(orig_img, cmap="gray")
        plt.arrow(x=3, y=3,
                  dx=_dx, dy=_dy,
                  color="red", head_width=0.5, head_length=0.3)
        plt.annotate(text="vector x (wavelength={:4.3f})".format(wavelength),
                     xy=(3, 3), xytext=((3 + _dx) / 2.0, (3 + _dy) / 2.0), fontsize=20, color="red")

        plt.title("$cos (-2\pi (  {}v/{} + {}u/{} ) )$".format(_y, _N, _x, _M), fontsize=30)
        plt.xlabel("$u$", fontsize=30)
        plt.ylabel("$v$", fontsize=30)
        plt.colorbar()
        plt.show()

def Fourier(_N,_M,_y,_x,orig_img):

    ySample, xSample = _N, _M
    fs = np.zeros((ySample, xSample), dtype=complex)
    startx = -orig_img.shape[1] / 2
    starty = -orig_img.shape[0] / 2
    countx = -1
    xs = range(int(startx), int(startx) + orig_img.shape[1])
    ys = range(int(starty), int(starty) + orig_img.shape[0])
    for x in xs:
        countx += 1
        county = -1
        for y in ys:
            county += 1
            fs[county, countx] = f(y, x, orig_img)
            if np.abs(fs[county, countx]) > 0.0001:
                print("x={:2.0f}, y={:2.0f}, f={:+5.3f}".format(x, y, fs[county, countx]))
    extent = (startx - 0.5,startx - 0.5 + orig_img.shape[1],starty - 0.5 + orig_img.shape[0],starty - 0.5)
    plt.figure(figsize=(20, 10))
    plt.imshow(np.abs(fs), cmap="gray", extent=extent)  # left, right, bottom, top)
    plot_nonzero_Fourier_coefficient_line(_y, _x, _N, _M)
    plt.xticks(xs)
    plt.yticks(ys)
    plt.xlabel("$u$", fontsize=30)
    plt.ylabel("$v$", fontsize=30)
    plt.title("DFT shape={}".format(fs.shape), fontsize=20)
    plt.colorbar()
    plt.show()

    #IDFT
    startx = - orig_img.shape[1] / 2
    starty = - orig_img.shape[0] / 2
    reco_img = np.zeros((ySample, xSample), dtype=complex)
    countx = -1
    for x in range(int(startx), int(startx) + orig_img.shape[1]):
        countx += 1
        county = -1
        for y in range(int(starty), int(starty) + orig_img.shape[0]):
            county += 1
            reco_img[county, countx] = F(y, x, fs)
    plt.imshow(np.real(reco_img), cmap="gray")
    plt.title("reconstructed image")

    plt.show()


def fft(_N,_M,_y,_x,orig_img):
    fft = np.fft.fft2(orig_img)
    startx = -orig_img.shape[1] / 2
    starty = -orig_img.shape[0] / 2
    extent = (startx - 0.5,
              startx - 0.5 + orig_img.shape[1],
              starty - 0.5 + orig_img.shape[0],
              starty - 0.5)
    fft_shift = np.fft.fftshift(fft)
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(np.abs(fft), cmap="gray")
    ax.set_title("FFT")
    ax.set_xlabel("$u$", fontsize=30)
    ax.set_ylabel("$v$", fontsize=30)

    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(np.abs(fft_shift), cmap="gray",
              extent=extent)  # (left, right, bottom, top
    ax.set_title("FFT f(0,0) at the center")
    ax.set_xlabel("$u$", fontsize=30)
    ax.set_ylabel("$v$", fontsize=30)
    plot_nonzero_Fourier_coefficient_line(_y, _x, _N, _M)
    plt.show()


def wavelengthImplement():
    _N = 18
    _M = 42
    _ys = [0, 3, 7, 5]
    _xs = [10, 0, 1, 4]

    for _y, _x in zip(_ys, _xs):
        dy = _y / float(_N)
        dx = _x / float(_M)
        orig_img = get_sinusoid_img(dy, dx, _M, _N)
        _dx, _dy, wavelength = get_wavelength_adjusted_vec(dx, dy)

        const = 0.5

        fig = plt.figure(figsize=(_M * const, _N * const))
        plt.imshow(orig_img, cmap="gray")
        plt.arrow(x=3, y=3,
                  dx=_dx, dy=_dy,
                  color="red", head_width=0.5, head_length=0.3)
        plt.annotate(text="vector x (wavelength={:4.3f})".format(wavelength),
                     xy=(3, 3), xytext=((3 + _dx) / 2.0, (3 + _dy) / 2.0), fontsize=20, color="red")
        plt.title("$cos (-2\pi (  {}v/{} + {}u/{} ) )$".format(_y, _N, _x, _M), fontsize=30)
        plt.xlabel("$u$", fontsize=30)
        plt.ylabel("$v$", fontsize=30)
        plt.colorbar()
        plt.show()


def FFT2Implement():
    _N = 18
    _M = 42
    _y = 5
    _x = 14
    dy = _y / float(_N)
    dx = _x / float(_M)
    orig_img = get_sinusoid_img(dy, dx, _M, _N)
    fft(_N, _M, _y, _x, orig_img)

def FourierImplement():
    _N = 18  ## the number of pixels along y axis
    _M = 42  ## the number of pixels along x axis
    _y = 5
    _x = 14
    dy = _y / float(_N)
    dx = _x / float(_M)
    orig_img = get_sinusoid_img(dy, dx, _M, _N)
    Fourier(_N, _M, _y, _x, orig_img)

if __name__ == '__main__':

    while True:
        print("1. Calculate the shortest wavelength")
        print("2. DFT and IDFT . Transforms")
        print("3. Fourier transform using python's ff2 library")
        user_input = input("Please choose from the following options:")

        try:
            number = int(user_input)
            if number == 1:
                wavelengthImplement()
            elif number == 2:
                FourierImplement()
            elif number == 3:
                FFT2Implement()
            else:
                print("The number you entered is not between 1 and 3. Please re-enter")

        except ValueError:
            print("You entered not an integer.Please re-enter")

