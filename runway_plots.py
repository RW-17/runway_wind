import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np

cmap = colormaps['YlOrRd']

# Dear user/programmer,
# When I wrote this code, only God and I knew how it worked.
# Now, only God knows it!

# Input data
theta = np.deg2rad(np.arange(0, 360, 22.5))
r = [2.5, 5.5, 9, 14, 19.5, 25, 31, 37.5]
idx = [[0, 4], [0, 5], [0, 6], [0, 6], [0, 7], [0, 4], [0, 5], [0, 6],
       [0, 8], [0, 7], [0, 8], [0, 6], [0, 5], [0, 4], [0, 4], [0, 4]]
r_lims = [0, 1, 4, 7, 11, 17, 22, 28, 34, 41]
prob = [[0.1, 1.0, 0.6, 0.2],
        [0.1, 1.6, 1.5, 0.8, 0.1],
        [0.1, 2.3, 3.1, 2.6, 0.5, 0.1],
        [0.1, 0.7, 0.9, 1.3, 0.4, 0.1],
        [0.1, 0.6, 0.9, 0.9, 0.2, 0.1, 0.1],
        [0.0, 0.3, 0.2, 0.2],
        [0.0, 0.4, 0.3, 0.3, 0.1],
        [0.0, 0.5, 0.4, 0.3, 0.1, 0.1],
        [0.2, 1.7, 1.5, 1.4, 0.6, 0.3, 0.1, 0.1],
        [0.1, 1.5, 1.8, 2.2, 1.0, 0.3, 0.1],
        [0.1, 2.8, 3.8, 3.8, 1.2, 0.4, 0.1, 0.1],
        [0.1, 1.1, 1.5, 1.2, 0.4, 0.1],
        [0.1, 1.3, 1.3, 0.7, 0.2],
        [0.0, 0.7, 0.6, 0.3],
        [0.0, 0.7, 0.6, 0.3],
        [0.0, 0.3, 0.1, 0.1],]

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
for i in range(16):
    theta_aux = theta[i] * np.ones([idx[i][1] - idx[i][0], 1])
    r_aux = r[idx[i][0]:idx[i][1]]
    ax.scatter(theta_aux, r_aux, marker='None')
    for j in range(len(theta_aux)):
        if prob[i][j] == 0:
            continue
        else:
            # ax.annotate("{}".format(prob[i][j]), xy=[theta_aux[j], r_aux[j]],
            #             ha='center', va='center')
            if prob[i][0] == 0:
                r_col = r_lims[2]
            else:
                r_col = r_lims[1]
            ax2 = ax.bar(theta[i], r_lims[j+2]-r_col, width=2*np.pi/16,
                         bottom=r_col, color=cmap((prob[i][j])/4.0),
                         edgecolor='gray', zorder=-j)

# norm = mpl.colors.Normalize(vmin=0, vmax=3.8)
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# fig.colorbar(sm, ticks=np.arange(0, 3.8, 0.1))
# Indication of calm status probability
# ax.annotate("{}".format(38.1), xy=[0, 0],
#             ha='center', va='center')

cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap),
                    ax=plt.gca(), pad=0.1)
cbar.set_ticks(ticks=np.linspace(0, 1, 11),
               labels=[round(0.1*x, 1) for x in range(0, 44, 4)])


def runway_plot(rwy, h):
    # factor_1 = 70/6
    factor_2 = 42/6
    ax.plot([rwy, rwy+np.pi], [41, 41], '-', c='blue', lw=h*factor_2,
            alpha=0.075)
    ax.plot([rwy, rwy+np.pi], [41, 41], '--', c='blue')
    ax.plot([-np.arcsin(h/41)+rwy, np.arcsin(h/41)+rwy+np.pi],
            [41, 41], c='blue')
    ax.plot([np.arcsin(h/41)+rwy, -np.arcsin(h/41)+rwy+np.pi],
            [41, 41], c='blue')
    ax.set_ylim(top=41)

    ax.set_yticklabels([])
    ax.set_yticks(r_lims)
    ax.set_xticks(theta)
    # ax.set_xticks(np.deg2rad(np.arange(11.25, 371.25, 22.5)))
    ax.set_xticklabels(['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S',
                        'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'])
    ax.set_theta_zero_location('N')

    ax.set_theta_direction(-1)
    ax.grid(False, axis='x')
    for k in np.arange(11.25, 191.25, 22.5):
        ax.plot(np.deg2rad([k, k+180]), [41, 41], '-',
                c='#b0b0b0', lw=0.8, zorder=-3)

    ax.xaxis.remove_overlapping_locs = True
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'runway_{h}.svg', bbox_inches='tight')


# Calculate wind speed sector areas
sectors = []
for i, r_lim in enumerate(r_lims[:-1]):
    sectors.append(np.deg2rad(11.25)*(r_lims[i+1]**2 - r_lim**2))
print('Area Check:', np.pi*41**2 - 16*np.sum(sectors))


def area_sector(phi: float, h: float, sector, r: float) -> float:
    if h > r:
        return 1.0

    phi = phi - sector
    theta = np.deg2rad(22.5)
    limit_2 = -np.arcsin(h/r) + (abs(phi)+theta/2)
    limit_3 = -np.arcsin(h/r) + (np.pi-abs(phi)+theta/2)
    if limit_2 <= 0 or limit_3 <= 0:
        return 1.0

    if abs(phi) <= np.pi/2:
        phi = abs(phi)
        limit = -np.arcsin(h/r) + (phi-theta/2)
        if limit > 0:
            area = 0.5*h*(h/np.tan(phi-theta/2)
                          - h*np.tan(np.pi/2-phi-theta/2))
            return area
        else:
            area = 0.5*((theta/2-phi+np.arcsin(h/r))*r**2
                        + (np.sqrt(r**2-h**2)
                        - h*np.tan(np.pi/2-phi-theta/2))*h)
            return area
    else:
        phi = np.pi - abs(phi)
        limit = -np.arcsin(h/r) + (phi-theta/2)
        if limit > 0:
            area = 0.5*h*(h/np.tan(phi-theta/2)
                          - h*np.tan(np.pi/2-phi-theta/2))
            return area
        else:
            area = 0.5*((theta/2-phi+np.arcsin(h/r))*r**2
                        + (np.sqrt(r**2-h**2)
                        - h*np.tan(np.pi/2-phi-theta/2))*h)
            return area


def calculate_sectors(phi: float, h: float):
    cover = []
    for sector in theta[0:8]:
        areas = []
        sec_areas = []
        for r in r_lims[1:]:
            areas.append(area_sector(phi, h, sector, r))
        for i, area in enumerate(areas):
            if area == 1.0:
                sec_areas.append(area)
            else:
                if areas[i-1] == 1.0:
                    sec_areas.append((area-np.sum(sectors[0:i]))/sectors[i])
                else:
                    sec_areas.append((area-areas[i-1])/sectors[i])
        cover.append(sec_areas[1:])
    cover.extend(cover)
    return cover


def percentages(phi: float, h: float):
    cover = calculate_sectors(phi, h)
    total_i = 38.1
    for i in range(len(theta)):
        limits = idx[i]
        for j in range(limits[0], limits[1]):
            # print(i, j, cover[i][j], prob[i][j])
            total_i += cover[i][j] * prob[i][j]
    return total_i


h = 20
wind_percents = []
for rwy in np.deg2rad(np.arange(0, 180, 2)):
    wind_percents.append(percentages(rwy, h))
    print(f"RWY {round(np.rad2deg(rwy))}, {percentages(rwy, h):3f}")

optimum = np.max(wind_percents)
rw_id = 2*np.argmax(wind_percents)
print(f"Optimal RWY: {rw_id}, Percentage: {optimum:.2f}")

rwy = np.deg2rad(rw_id)
runway_plot(rwy, h)
