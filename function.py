from BeamCalc import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import io
import base64


def caller(input):
    
    ss = SystemElements(EA=15000, EI=5000)
    variables = []
    variables.append(float(input['BeamLen']))

    for each in input['Pin']:
        variables.append(float(each))

    for each in input['Roller']:
        variables.append(float(each))

    for each in input['Fixed']:
        variables.append(float(each))

    for each in input['PL']:
        variables.append(float(each['location']))

    for each in input['Mom']:
        variables.append(float(each['location']))

    for each in input['UDL']:
        variables.append(float(each['start']))
        variables.append(float(each['end']))
    

    variables.sort()
    nodes = list(set(variables))
    nodes.sort()
    if nodes[0] != 0:
        nodes.insert(0,float(0))

    for each in nodes:
        if each > 0:
            ss.add_element([each, 0])
            print("add element")
            print(each)

    for each in input['Pin']:
        nodePin = int(nodes.index(float(each)) + 1)
        ss.add_support_hinged([nodePin])
        print("hing")
        print(nodePin)

    for each in input['Fixed']:
        nodeFixed = int(nodes.index(float(each)) + 1)
        ss.add_support_fixed([nodeFixed])
        print("supportfix")
        print(nodeFixed)

    for each in input['Roller']:
        nodeRoller = int(nodes.index(float(each)) + 1)
        ss.add_support_roll(nodeRoller, direction='x')
        print("supportRoller")
        print(nodeRoller)

    for each in input['PL']:
        locPL = float(each['location'])
        nodePL = int(nodes.index(locPL) + 1)
        loadPL = -1 * float(each['load'])
        ss.point_load(nodePL, Fx=0, Fy = loadPL)
        print("PL")
        print(nodePL)
        print(loadPL)

    for each in input['Mom']:
        locMom = float(each['location'])
        nodeMom = int(nodes.index(locMom) + 1)
        loadMom = float(each['load'])
        ss.moment_load(nodeMom, Ty= loadMom)
        print("Mom")
        print(nodeMom)

    for each in input['UDL']:
        startUDL = float(each['start'])
        endUDL = float(each['end'])
        nodeStart = int(nodes.index(startUDL) + 1)
        nodeEnd = int(nodes.index(endUDL) + 1)
        for i in range(nodeStart, nodeEnd):
            ss.q_load(-1 * float(each['load']),[i])
            print("UDL")
    
    ss.solve()

    x ,y = ss.show_bending_moment(show = False, values_only = True, factor = 1)
    neg = -1
    y = neg * np.array(y)
    maxbm = float(max(y))
    plt.axhline(linewidth=1, color='k')
    plt.plot(x,y)

    plt.title('Bending Moment')
    plt.xlabel('Length (m)')
    plt.ylabel('Bending Moment (Nm)')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    bm = base64.b64encode(img.getvalue()).decode('utf8')
    plt.figure()

    x ,y = ss.show_shear_force(show = False, values_only = True, factor = 1)
    neg = -1
    y = neg * np.array(y)
    maxshear = float(max(y))
    plt.axhline(linewidth=1, color='k')
    plt.plot(x,y)

    plt.title('Shear Force')
    plt.xlabel('Length (m)')
    plt.ylabel('Shear Force (N)')
    imgShear = io.BytesIO()
    plt.savefig(imgShear, format='png')
    imgShear.seek(0)
    shear = base64.b64encode(imgShear.getvalue()).decode('utf8')
    plt.figure()

    graphs = [bm, shear, maxbm, maxshear]

    return graphs