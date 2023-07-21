import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy
from matplotlib import cm


main_path = "./data/Illumination_Homogeneity"
folder_ninety = "/ImgProj_bright_90Deg"
folder_zero = "/ImgProj_bright_0Deg"
folder_prev = "/ImgProj_bright"
img_name = "/Ultrasat_BSI_L_T112181_W06_D02_ImgProj_bright_0001.tif"
dark_img_path = "/Ultrasat_BSI_L_T112181_W06_D02_ImgProj_dark_0001.tif"

all_path1 = main_path + folder_ninety + img_name
all_path2 = main_path + folder_zero + img_name
prev_all_path = main_path + folder_prev + img_name
dark_path = main_path + dark_img_path


def get_image(path, gain):
    im = Image.open(path)
    # find th position to crop the images
    img_l = np.array(im.crop((0, 0, 4742, 4742)))
    img_h = np.array(im.crop((4742, 0, 9484, 4742)))
    # img = np.add(img, img_1)
    m, n = img_h.shape
    img_lg = img_l[4 : m - 4, 4 : n - 4]
    img_hg = img_h[4 : m - 4, 4 : n - 4]
    if gain == "low":
        return img_lg
    elif gain == "high":
        return img_hg


# Add function for showing the image in required level
def save_image(img, path_name, img_title):
    # dx, dy = 1000, 1000
    print(img_title, np.mean(img))
    # img = (img/np.mean(img))
    mean_img = np.mean(img)
    std_img = np.std(img)
    # img[:, ::dy] = 0
    # img[::dx, :] = 0
    plt.title(path_name + "\n" + img_title)
    plt.imshow(img[2000:3000, 2000:3000], cmap="jet")
    plt.colorbar()
    # plt.clim(0.94, 1.06)
    plt.clim(mean_img - 4 * std_img, mean_img + 4 * std_img)
    plt.tight_layout()
    img_name = path_name + "/" + img_title + ".png"
    plt.savefig(img_name)
    # plt.show()
    plt.close()


def get_mean_hist(img1, img2, img3):
    mean1 = np.mean(img1)
    # mean2 = np.mean(img2)
    # mean3 = np.mean(img3)
    # img1 = np.subtract(img1, mean1)
    # img2 = np.subtract(img2, mean2)
    # img3 = np.subtract(img3, mean3)
    m1 = np.mean(img1)
    s1 = np.std(img1)
    # m2 = np.mean(img2)
    # s2 = np.std(img2)
    # m3 = np.mean(img3)
    # s3 = np.std(img3)
    # mean_img = np.mean(img, axis=0)
    print("Standard deviations are", s1)
    print("mean values are", mean1)
    plt.hist(
        img1.flatten(), range=[m1 - 4 * s1, m1 + 4 * s1], alpha=0.5, label="Previous"
    )
    # plt.hist(img2.flatten(), range=[m2 - 4 * s2, m2 + 4 * s2], alpha=0.5, label='0 degree')
    # plt.hist(img3.flatten(), range=[m3 - 4 * s3, m3 + 4 * s3], alpha=0.5, label='90 degrees')
    plt.xlabel("Pixel signal value", fontsize=14)
    plt.ylabel("Number of pixels", fontsize=14)
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    # plt.hist(mean_img.flatten(), 1000, range=[12000, 14000], label="HG", alpha=0.5)
    # plt.show()


def plot_histogram(img1, label):
    mean1 = np.round(np.mean(img1), 3)
    # img1 = np.subtract(img1, mean1)
    m1 = np.mean(img1)
    s1 = np.round(np.std(img1), 2)
    print("Standard deviation is", s1)
    print("mean is", mean1)
    plt.hist(
        img1.flatten(),
        bins = np.arange(m1 - 4*s1, m1 + 4*s1, 1),
        alpha=0.5,
        label="mean = " + str(mean1) + "\n std =" + str(s1),
    )
    plt.xlabel("Pixel signal value", fontsize=14)
    plt.ylabel("Number of pixels", fontsize=14)
    plt.title(label)
    plt.legend()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    name = main_path + '/' + label + '.png'
    plt.savefig(name)
    # plt.show()
    plt.close()


def get_mean_sections(img):
    mean_sec = []
    std_sec = []
    c, r = img.shape
    for i in range(0, 4734, 1000):
        for j in range(0, 4734, 1000):
            img_sec = img[i : i + 1000, j : j + 1000]
            # print(img_sec.shape)
            mean_img = np.mean(img_sec)
            std_img = np.mean(img_sec)
            m_img = np.round(mean_img, 2)
            s_img = np.round(std_img, 2)
            mean_sec.append(m_img)
            std_sec.append(s_img)
    return mean_sec, std_sec


def get_corrected_img(path, gain):
    img = get_image(path, gain)
    img = img.astype(np.int16, casting="same_kind")
    dark = get_image(dark_path, gain)
    dark = dark.astype(np.int16, casting="same_kind")
    corr_img = np.subtract(img, dark)
    return corr_img


def find_structures(img1, img2):
    img1 = np.abs(img1)+1
    img2 = np.abs(img2) + 1
    img1 = img1/np.mean(img1)
    img2 = img2/np.mean(img2)
    divided_img = np.divide(img1, img2)
    save_image(divided_img, main_path, "divided image")
    return 0


def find_std(img, mean_v):
    img_bs = np.subtract(img, mean_v)
    img_sq = np.multiply(img_bs, img_bs)
    img_var = np.mean(img_sq)
    print('Variance of the image is', img_var)
    img_std = np.sqrt(img_var)
    per_dev = (img_std*100/mean_v)
    print('Image std and % deviation', img_std, per_dev)
    return img_std, per_dev


def plot_1D_profile(img):
    m, n = img.shape
    img_hor = img[(m//2)-50: (m//2)+50, :]
    img_ver = img[:, (m//2)-50: (m//2)+50]
    avg_img_hor = np.mean(img_hor, axis=0)
    print(avg_img_hor.shape)
    cols = np.arange(0, len(avg_img_hor))
    avg_img_ver = np.mean(img_ver, axis=1)
    rows = np.arange(0, len(avg_img_ver))
    print(avg_img_ver.shape)
    # Find a curve that fit the surface
    p_rows = np.polyfit(rows, avg_img_hor, 2)
    p_cols = np.polyfit(cols, avg_img_ver, 2)
    residue_rows = 100*(avg_img_hor-np.polyval(p_rows, rows))/avg_img_hor
    residue_cols = 100*(avg_img_ver - np.polyval(p_cols, rows)) / avg_img_hor
    print('row line profile', p_rows)
    print('Column line profile', p_cols)
    # column numbers
    plt.subplot(2, 2, 1)
    plt.scatter(cols, avg_img_ver, alpha=0.25, label='Pixel signal level')
    plt.plot(cols, np.polyval(p_cols, cols), color='orange', label='2nd degree polynomial fit')
    plt.xlabel('Pixel column number', fontsize=14)
    plt.ylabel('Intensity value (ADU)', fontsize=14)
    plt.title('Horizontal profile', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.scatter(cols, residue_cols, alpha=0.25)
    plt.xlabel('Pixel column number', fontsize=14)
    plt.ylabel('% Residue', fontsize=14)
    plt.title('Residue Horizontal profile', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.subplot(2, 2, 3)
    plt.scatter(rows, avg_img_hor,  alpha=0.25, label='Pixel signal level')
    plt.plot(rows, np.polyval(p_rows, rows), color='orange', label='2nd degree polynomial fit')
    plt.xlabel('Pixel row number', fontsize=14)
    plt.ylabel('Intensity value (ADU)', fontsize=14)
    plt.title('Vertical profile', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.scatter(rows, residue_rows, alpha=0.25)
    plt.xlabel('Pixel row number', fontsize=14)
    plt.ylabel('% Residue', fontsize=14)
    plt.title('Residue Vertical profile', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()
    # Fit the surface here.


def plot_2D_profile(img):
    # plot the 3D profile of the beam.
    m, n = img.shape
    y = np.arange(0, m, 1)
    x = np.arange(0, n, 1)
    xx, yy = np.meshgrid(x, y)
    # Flatten all the arrays
    X = xx.flatten()
    Y = yy.flatten()
    Z = img.flatten()
    data = np.c_[X, Y, Z]

    # best-fit linear plane (1st-order)
    # A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
    # C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])  # coefficients
    # evaluate it on grid
    # Z = C[0] * X + C[1] * Y + C[2]
    # Fitting a quadratic surface
    A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2]
    C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])
    # Subtract the residue from the image.
    print('Polynomials', C)
    check_arr = np.c_[np.ones(X.shape), X, Y, X * Y, X ** 2, Y ** 2]
    print('Check array shape', check_arr.shape)
    print('C shape', C.shape)
    # evaluate it on a grid
    Z = np.dot(np.c_[np.ones(X.shape), X, Y, X * Y, X ** 2, Y ** 2], C).reshape(X.shape)
    print(Z.shape)
    # Reshaped array
    Reshape = Z.reshape(m, n)
    print(Reshape.shape)
    difference = img - Reshape
    print(np.mean(difference))
    mean_arr = np.mean(img)
    std_arr = np.std(img)
    mean_diff = np.mean(difference)
    std_diff = np.std(difference)
    # fit the surface here
    fig1 = plt.figure()
    ax = fig1.add_subplot(projection='3d')
    # surf = ax.plot_surface(xx, yy, img,  cmap=cm.coolwarm) # 'plasma')
    surf = ax.plot_surface(xx, yy, Reshape, cmap='plasma')
    ax.set_zlim(mean_arr-2*std_arr, mean_arr+2*std_arr)
    # Add a color bar which maps values to colors.
    fig1.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    fig2 = plt.figure()
    ax1 = fig2.add_subplot(projection='3d')
    surf1 = ax1.plot_surface(xx, yy, difference, cmap='plasma')
    ax1.set_zlim(mean_diff-2*std_diff, mean_diff+2*std_diff)
    fig2.colorbar(surf1, shrink=0.5, aspect=5)
    plt.show()
    fig3 = plt.figure()
    mean_diff = np.mean(difference)
    std_diff = np.std(difference)
    plt.imshow(difference, cmap='jet')
    plt.clim(mean_diff - std_diff, mean_diff + std_diff)
    plt.colorbar()
    plt.show()
    # divide the img array with the surface.
    divided_surf = np.divide(img, Reshape)
    mean_div = np.mean(divided_surf)
    print(mean_div)
    std_div = np.std(divided_surf)
    print(std_div)
    plt.imshow(divided_surf)
    plt.clim(-2, 2)
    plt.colorbar()
    plt.show()



