"""
SimitateTrajectoryLoader
Usage like:

    import trajectory_loader as tl
    a = tl.SimitateTrajectoryLoader()
    # load trajectory
    a.load_trajectories("./generated_trajectories/heart_2018-09-17-21-52-09.csv",["hand"])
    # save plot as pdf
    a.plot_trajectories_to_file("test.pdf")

"""

import pickle
import numpy as np
from tf import transformations
import csv
#import io



def filter_lines(f, target_frame):
    #result = []
    for i, line in enumerate(f):
        elements = line.split(",")
        if elements[3] == "world" and elements[4] == target_frame:
            # print(elements)
            yield line
            #result.append(line)
    #return result


def own_genfromtxt(filtered_lines):
    for line in filtered_lines:
        line.split(",")


def matrix_from_row(row):
    translation = row[5:8]
    quaternion = row[8:12]
    mat_t = transformations.translation_matrix(translation)
    mat_q = transformations.quaternion_matrix(quaternion)
    return mat_t.dot(mat_q)



class SimitateTrajectoryLoader(object):
    """docstring for SimitateTrajectoryLoader"""

    def __init__(self):
        super(SimitateTrajectoryLoader, self).__init__()
        self.trajectories = {}

    def get_frames(self, filename):
        frame_blacklist = ["field.transforms0.child_frame_id", "kinect2"]
        frames = []
        input_file = open(filename)
        for i, line in enumerate(input_file):
            elements = line.split(",")
            important_element = elements[4]
            if important_element in frames:
                continue
            for black_frame in frame_blacklist:
                if black_frame in important_element:
                    break
            else:
                print ("    found frame " + str(important_element))
                frames.append(str(important_element))
        input_file.close()
        return frames


    def get_transform_to_world(self):
        mat_w_2_mocap = transformations.identity_matrix()
        exist_mat_w_2_mocap = False
        mat_mocap_2_link = transformations.identity_matrix()
        exists_mat_mocap_2_link = False
        mat_link_2_rgb = transformations.identity_matrix()
        exists_mat_link_2_rgb = False
        mat_optical = transformations.identity_matrix()
        exists_mat_optical = False
        with open(self.filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if (not exist_mat_w_2_mocap) and (row[3] == "world" and row[4] == "kinect2_mocap"):
                    exist_mat_w_2_mocap = True
                    mat_w_2_mocap = matrix_from_row(row)
                if (not exists_mat_mocap_2_link) and (row[3] == "/kinect2_mocap" and row[4] == "/kinect2_link"):
                    exists_mat_mocap_2_link = True
                    mat_mocap_2_link = matrix_from_row(row)
                if (not exists_mat_link_2_rgb) and (row[3] == "kinect2_link" and row[4] == "kinect2_rgb_optical_frame"):
                    exists_mat_link_2_rgb = True
                    mat_link_2_rgb = matrix_from_row(row)
                if (not exists_mat_optical) and (row[3] == "kinect2_rgb_optical_frame" and row[4] == "kinect2_ir_optical_frame"):
                    exists_mat_optical  = True
                    mat_optical = matrix_from_row(row)
                if exist_mat_w_2_mocap and exists_mat_mocap_2_link and exists_mat_link_2_rgb and exists_mat_optical:
                    break
            else:
                print ("Warning! No transformation in world found")
        # print mat_w_2_mocap, mat_mocap_2_link, mat_link_2_rgb
        # return mat_mocap_2_link.dot(transformations.inverse_matrix(mat_link_2_rgb.dot(mat_w_2_mocap)))
        return mat_w_2_mocap.dot(mat_mocap_2_link.dot(mat_link_2_rgb.dot(mat_optical)))
        # return transformations.inverse_matrix(mat_w_2_mocap.dot(mat_mocap_2_link.dot(mat_link_2_rgb)))

    def transform_to_world(self, point):
        return self.transform_to_world_matrix.dot([point[0],
                                                   point[1],
                                                   point[2],
                                                   1])

    def load_trajectories(self, filename, frames=None, with_orientation=False):

        """ this method loads dumped tf data from a csv file
        this file can be created by the following command:
            `rostopic echo -b <bag_file> -p /tf >> tf.csv`

        The loaded trajectories are stored in self.trajectories

        :filename: String of the csv file
        :frames: array of Strings of the frames to be loaded
        :returns: None

        """
        self.filename = filename
        if frames is None:
            frames = self.get_frames(filename)
            print("Found the following frames: ", frames)
        data = self.trajectories
        for current_frame in frames:
            #stream = io.open(filename,'rb') didnot work
            with open(filename) as f:
                if with_orientation:
                    cols = (2, 5, 6, 7, 8, 9, 10, 11)
                else:
                    cols = (2, 5, 6, 7)
                current_trajectory = np.genfromtxt(filter_lines(
                                                   f, current_frame),
                                                   dtype=np.float64,
                                                   delimiter=",",
                                                   # usecols=(2, 5, 6, 7, 8, 9, 10, 11))
                                                   usecols=cols)
                # convert to dict from timestamp to 3d points
                # timestamp_to_trajectory = dict(zip(current_trajectory[:, 0], current_trajectory[:, 1:]))
                data[current_frame] = current_trajectory #timestamp_to_trajectory
        self.trajectories = data
        self.transform_to_world_matrix = self.get_transform_to_world()

    def add_point_to_trajectory(self, frame, stamp, point=[0, 0, 0]):
        """This methods adds trajectory points to a given trajectory

        :frame: name of the trajectory, needs to be for later access and is also
                used for plotting
        :stamp: timestamp, serves as index for looking up the the trajectory,
                if no time is available, you can use a index
        :point: the actual trajectory point in 3D as list
        :returns: None

        """


        if self.trajectories is None:
            print("No trajectory existing yet, will create one with frame %s " % frame)
            self.trajectories = {}
            self.trajectories[frame] = {}
        if frame not in self.trajectories:
            self.trajectories[frame] = {}
            a = np.array([[str(stamp), point[0], point[1], point[2]]], np.float64)
            self.trajectories[frame] = a
        else:
            self.trajectories[frame] = np.append(self.trajectories[frame], [[stamp, point[0], point[1], point[2]]], axis=0)
        # self.trajectories[frame][stamp] = point

    def plot_trajectories(self):
        """This method will plot all trajectories loaded and show the plot
        on screen for saving the plot use the method `plot_trajectories_to_file`
        """
        # plt.title()
        # ax = fig.add_subplot()
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d
        from matplotlib import cm

        fig = plt.figure()

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        ax = fig.add_subplot("111", projection='3d')
        for frame in self.trajectories:
            # data = np.array(list(self.trajectories[frame].values()))
            data = self.trajectories[frame]
            # ax.plot(data[:1000, 0], data[:1000, 1], data[:1000, 2], label=frame)
            frame = frame.replace("_", "-")
            if frame == "baseline_trajectory":
                ax.plot(data[:, 1], data[:, 2], data[:, 3], "ro", label=frame)
            else:
                ax.plot(data[:, 1], data[:, 2], data[:, 3], label=frame)
            #ax.plot(data[:, 0], data[:, 1], np.zero(len(data)), label=frame)
            # 2d
            #plt.plot(data[:, 0], data[:, 2], 'ro')
        ax.legend()
        #plt.axis([-1, 1, -1, 1])
        plt.show()
        return fig
        #dest_filename = os.path.splitext(os.path.basename(sys.argv[1]))[0]+".pdf"
        #plt.savefig(dest_filename, bbox_inches='tight')

    def plot_trajectories_to_file(self, filename):
        """
        plots the trajectories into a file depending on the extention

        :filename: resulting filename where the resulting pdf is saved to
        """
        fig = self.plot_trajectories()
        dest_filename = filename
        fig.savefig(dest_filename, bbox_inches='tight')

    def save_trajectory_to_file(self, filename):
        with open(filename, 'w') as f:  # Python 3: open(..., 'wb')
            pickle.dump(self.trajectories, f)

    def load_trajectories_from_file(self, filename):
        with open(filename, 'rb') as f:  # Python 3: open(..., 'rb')
            self.trajectories = pickle.load(f)

    def animate_trajectories(self, frame):
        from matplotlib.animation import FuncAnimation

        # def update_lines(num, dataLines):
            # set_data(data[0:2, :num])
            # line.set_3d_properties(data[2, :num])
            # return lines

        # Attaching 3D axis to the figure
        fig = plt.figure()
        # ax = p3.Axes3D(fig)
        ax = fig.add_subplot(111, projection='3d')

        # def gen(n):
            # phi = 0
            # while phi < 2*np.pi:
                # yield np.array([np.cos(phi), np.sin(phi), phi])
                # phi += 2*np.pi/n

        def update(num, data, line):
            # print(data[:num, 1], data[:num, 2])
            print(num)
            # line.set_data(data[:num, 1], data[:num, 2]) #, data[:1, 2)
            # line.set_3d_properties(0)
            # line.set_3d_properties(data[:num, 3])
            # line.set_data(data[:2, :num].T)
            # line.set_3d_properties(data[:num, 2])

        data = np.array(list(self.trajectories[frame].values())) #np.array(list(gen(N))).T
        print(data)
        print(data.shape)
        # line, = ax.plot(data[:, 1], data[:, 2])

        # data = 
        line, = ax.plot(data[:, 1], data[:, 2], data[:, 2])
        # line, = ax.plot(data[0])
        
        # Setting the axes properties
        ax.set_xlim3d([-1.0, 1.0])
        ax.set_xlabel('X')
        
        ax.set_ylim3d([-1.0, 1.0])
        ax.set_ylabel('Y')
        
        ax.set_zlim3d([0.0, 10.0])
        ax.set_zlabel('Z')
        
        ani = animation.FuncAnimation(fig, update, len(self.trajectories[frame]), fargs=(data, line), interval=1, blit=False)
        #ani.save('matplot003.gif', writer='imagemagick')
        plt.show()
        
        # # Fifty lines of random 3-D lines
        # # data = [Gen_RandLine(25, 3) for index in range(50)]
        # data = np.array(list(self.trajectories[frame].values()))
        
        # # Creating fifty line objects.
        # print(data)
        # # NOTE: Can't pass empty arrays into 3d version of plot()
        # lines = [ax.plot(dat[0], dat[1], dat[2]) for dat in data]
        
        # # Setting the axes properties
        # ax.set_xlim3d([0.0, 1.0])
        # ax.set_xlabel('X')
        
        # ax.set_ylim3d([0.0, 1.0])
        # ax.set_ylabel('Y')
        
        # ax.set_zlim3d([0.0, 1.0])
        # ax.set_zlabel('Z')
        
        # ax.set_title('3D Test')
        
        # # Creating the Animation object
        # line_ani = FuncAnimation(fig, update_lines, len(data), fargs=(data),
                                           # interval=50, blit=False)
        
        # plt.show()
