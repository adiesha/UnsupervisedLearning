import random
import pandas as pd
import matplotlib.pyplot as plt
from generate_polygon import polygon_shape

random.seed(42)


class DataWithLabel:

    def __init__(self, num_poly, x_range, y_range, num_points):
        self.data = []
        self.label = []
        self.num_poly = num_poly
        self.x_range = x_range
        self.y_range = y_range
        self.num_points = num_points

    def artificial_data(self):
        
        polygons, centers, radius = polygon_shape(self.num_poly, self.x_range, self.y_range)

        xmin = []
        xmax = []
        ymin = []
        ymax = []
        
        for pt in range(len(centers)):
            tmp = centers[pt]
            xmin.append(tmp[0] - radius)
            xmax.append(tmp[0] + radius)
            ymin.append(tmp[1] - radius)
            ymax.append(tmp[1] + radius)
            
        xmin = [self.x_range[0] if i <= self.x_range[0] else i for i in xmin]
        xmax = [self.x_range[1] if i >= self.x_range[1] else i for i in xmax]
        ymin = [self.y_range[0] if i <= self.y_range[0] else i for i in ymin]
        ymax = [self.y_range[1] if i >= self.y_range[1] else i for i in ymax]
            
        estimate_points = int(self.num_points / (self.num_poly + 1))
        noise_points = self.num_points - (estimate_points * self.num_poly)
        
        for poly in range(len(polygons)):
            
            tmp_x_range = [int(xmin[poly]), int(xmax[poly] + 1)]
            tmp_y_range = [int(ymin[poly]), int(ymax[poly] + 1)]
            
            i = 0
            while i < estimate_points:
                x = random.randrange(*tmp_x_range)
                y = random.randrange(*tmp_y_range)
                
                if polygons[poly].contains_point([x, y]):
                    self.data.append((x, y))
                    self.label.append(poly + 1)
                    i += 1
                    
        i = 0
        while i < noise_points:
            x = random.randrange(*self.x_range)
            y = random.randrange(*self.y_range)
            self.data.append((x, y))
            self.label.append(0)
            i += 1
            
        noise_start_idx = estimate_points * self.num_poly - 1
        
        for tmp in range(noise_start_idx, self.num_points - 1):
            for poly in range(len(polygons)):
                if polygons[poly].contains_point(self.data[tmp]):
                    self.label[tmp] = poly
            
        return polygons


abc = DataWithLabel(6, [0, 500], [0, 500], 5000)
polygon_all = abc.artificial_data()
check = abc.data
val = abc.label

# save the data and labels
df_data = pd.DataFrame(check)
df_label = pd.DataFrame(val)
df_data.to_csv('Synthetic_Values.csv', index = False, header=False)
df_label.to_csv('Synthetic_Values_Labels.csv', index = False, header=False)

# display the figure, and save it
plt.axes()
x_points = []
y_points = []
for points in check:
    x_points.append(points[0])
    y_points.append(points[1])
plt.plot(x_points, y_points, 'ro')
for poly in polygon_all:
    points = poly
    polygon = plt.Polygon(points)
    plt.gca().add_patch(polygon)

plt.axis('scaled')
plt.show()
plt.savefig('Synthetic_Data_Image.png')
