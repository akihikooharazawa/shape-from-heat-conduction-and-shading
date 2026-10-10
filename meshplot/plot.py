from .Viewer import Viewer
import numpy as np
from ipywidgets import Output, HBox, VBox
from ipywidgets import interact, widgets
# from IPython.display import display
import uuid

rendertype = "JUPYTER" # "OFFLINE"
def jupyter():
    global rendertype
    rendertype = "JUPYTER"

def offline():
    global rendertype
    rendertype = "OFFLINE"

def website():
    global rendertype
    rendertype = "WEBSITE"

class Subplot():
    def __init__(self, data, view, s, label=None):

        if data == None:
            self.render_outs = []
            self.hboxes = []
        else:
            self.render_outs = data.render_outs

        # if s[0] != 1 or s[1] != 1:
        if data == None: # Intialize subplot array
            cnt = 0
            for r in range(s[0]):
                row = []
                for c in range(s[1]):
                    row.append(Output())
                    cnt += 1
                self.render_outs.append(row)

            for r in self.render_outs:
                hbox = HBox(r)
                if rendertype == "JUPYTER":
                    display(hbox)
                self.hboxes.append(hbox)

        out = self.render_outs[int(s[2]/s[1])][s[2]%s[1]]
        if rendertype == "JUPYTER":
            # with out:
            dis_all = [view._renderer]
            if label is not None:
                dis_all.append(widgets.Label(value=str(label)))
            dis = VBox(dis_all)
            display(dis)
            # display(view._renderer)
        self.render_outs[int(s[2]/s[1])][s[2]%s[1]] = view

    def update_object(self, s, oid=0, v=None, f=None, c=None):
        return self.render_outs[int(s[2]/s[1])][s[2]%s[1]].update_object(oid, v, c, f)

    def add_lines_to_subplot(self, s, beginning, ending, shading={}, obj=None, **kwargs):
        return self.render_outs[int(s[2]/s[1])][s[2]%s[1]].add_lines(beginning, ending, shading=shading, obj=obj, **kwargs)
    
    def remove_object(self, s, obj_id):
        return self.render_outs[int(s[2]/s[1])][s[2]%s[1]].remove_object(obj_id)

    def remove_object_type(self, s, obj_type):
        return self.render_outs[int(s[2]/s[1])][s[2]%s[1]].remove_object_type(obj_type)
    
    def rotate_camera(self, s, azimuth=0.0, elevation=0.0):
        return self.render_outs[int(s[2]/s[1])][s[2]%s[1]].rotate_camera(azimuth, elevation)
    
    def get_renderer_stream(self, s):
        return self.render_outs[int(s[2]/s[1])][s[2]%s[1]].get_renderer_stream()
    
    def save(self, filename=""):
        if filename == "":
            uid = str(uuid.uuid4()) + ".html"
        else:
            filename = filename.replace(".html", "")
            uid = filename + '.html'

        s = ""
        imports = True
        for r in self.render_outs:
            for v in r:
                s1 = v.to_html(imports=imports, html_frame=False)
                s = s + s1
                imports = False

        s = "<html>\n<body>\n" + s + "\n</body>\n</html>"
        with open(uid, "w") as f:
            f.write(s)
        print("Plot saved to file %s."%uid)

    def to_html(self, imports=True, html_frame=True):
        s = ""
        for r in self.render_outs:
            for v in r:
                s1 = v.to_html(imports=imports, html_frame=html_frame)
                s = s + s1
                imports = False

        return s

def plot(v, f=None, c=None, uv=None, n=None, shading={}, plot=None, return_plot=True, filename="", texture_data=None, **kwargs):#, return_id=False):
    shading.update(kwargs)
    if not plot:
        view = Viewer(shading)
    else:
        view = plot
        view.reset()
    if type(f) == type(None): # Plot pointcloud
        obj_id = view.add_points(v, c, shading=shading)
    elif type(f) == np.ndarray and len(f.shape) == 2 and f.shape[1] == 2: # Plot edges
        obj_id = view.add_edges(v, f, shading=shading)
    else: # Plot mesh
        obj_id = view.add_mesh(v, f, c, uv=uv, n=n, shading=shading, texture_data=texture_data)

    if not plot and rendertype == "JUPYTER":
        dis_all = [view._renderer]
        if filename != "":
            dis_all.append(widgets.Label(value=str(filename)))
        dis = VBox(dis_all)
        display(dis)
        # display(view._renderer)
        

    if rendertype == "OFFLINE":
        view.save(filename)

    if return_plot or rendertype == "WEBSITE":
        return view

def subplot(v, f=None, c=None, uv=None, n=None, label=None, shading={}, s=[1, 1, 0], data=None, texture_data=None, **kwargs):
    shading.update(kwargs)
    shading["width"] = 400
    shading["height"] = 400
    view = Viewer(shading)
    if type(f) == type(None): # Plot pointcloud
        obj_id = view.add_points(v, c, shading=shading)
    elif type(f) == np.ndarray and len(f.shape) == 2 and f.shape[1] == 2: # Plot edges
        obj_id = view.add_edges(v, f, shading=shading)
    else: # Plot mesh
        obj_id = view.add_mesh(v, f, c, uv=uv, n=n, shading=shading, texture_data=texture_data)

    subplot = Subplot(data, view, s, label=label)
    if data == None or rendertype == "WEBSITE":
        return subplot
