#! /usr/bin/env python



import os,sys
import numpy as np
import xml.etree.ElementTree as ET


def formatFile(ifile,ofile=None):
    import shutil
    tmpfile = ifile.replace(".xml","_copy.xml")
    if ofile is not None:
        cmd = "xmllint --format "+ifile+" > "+ofile
    else:
        cmd = "xmllint --format "+ifile+" > "+tmpfile
    os.system(cmd)
    if ofile is None:
        shutil.move(tmpfile,ifile)
    return

millennium = False
millennium2 = False
millgas = True

if millennium:
    root = ET.Element("simulation")
    root.set("name","Millennium")
    COS = ET.SubElement(root,"cosmology")
    ET.SubElement(COS,"OmegaM").text = "0.25"
    ET.SubElement(COS,"OmegaL").text = "0.75"
    ET.SubElement(COS,"OmegaB").text = "0.045"
    ET.SubElement(COS,"H0").text = "73"
    ET.SubElement(COS,"sigma8").text = "0.9"
    ET.SubElement(COS,"ns").text = "1.0"
    part = ET.SubElement(root,"particles")
    mass = ET.SubElement(part,"mass")
    mass.text = "8.606567e8"
    mass.set("units","Msol/h")
    ET.SubElement(part,"number").text = str(2160**3)
    part = ET.SubElement(root,"boxSize")
    part.text = "500.0"
    part.set("units","Mpc/h")
    zfile = "/Users/amerson/Data/Simulations/millennium/redshift_list"
    dtype = [("iz",int),("z",float)]
    zdata = np.loadtxt(zfile,usecols=[0,1],dtype=dtype,skiprows=1).view(np.recarray)
    snaps = ET.SubElement(root,"snapshots")
    for i,z in zip(zdata.iz,zdata.z):
        snap = ET.SubElement(snaps,"snapshot")
        snap.text = str(z)
        snap.set("number",str(i))
    tree = ET.ElementTree(root)
    tree.write("millennium.xml")
    formatFile("millennium.xml")



if millennium2:
    root = ET.Element("simulation")
    root.set("name","Millennium2")
    COS = ET.SubElement(root,"cosmology")
    ET.SubElement(COS,"OmegaM").text = "0.25"
    ET.SubElement(COS,"OmegaL").text = "0.75"
    ET.SubElement(COS,"OmegaB").text = "0.045"
    ET.SubElement(COS,"H0").text = "73"
    ET.SubElement(COS,"sigma8").text = "0.9"
    ET.SubElement(COS,"ns").text = "1.0"
    part = ET.SubElement(root,"particles")
    mass = ET.SubElement(part,"mass")
    mass.text = "6.885e6"
    mass.set("units","Msol/h")
    ET.SubElement(part,"number").text = str(2160**3)
    part = ET.SubElement(root,"boxSize")
    part.text = "100.0"
    part.set("units","Mpc/h")
    zfile = "/Users/amerson/Data/Simulations/millennium2/redshift_list"
    dtype = [("iz",int),("z",float)]
    zdata = np.loadtxt(zfile,usecols=[0,1],dtype=dtype).view(np.recarray)
    snaps = ET.SubElement(root,"snapshots")
    for i,z in zip(zdata.iz,zdata.z):
        snap = ET.SubElement(snaps,"snapshot")
        snap.text = str(z)
        snap.set("number",str(i))
    tree = ET.ElementTree(root)
    tree.write("millennium2.xml")
    formatFile("millennium2.xml")



if millgas:
    root = ET.Element("simulation")
    root.set("name","MS-W7")
    COS = ET.SubElement(root,"cosmology")
    ET.SubElement(COS,"OmegaM").text = "0.272"
    ET.SubElement(COS,"OmegaL").text = "0.728"
    ET.SubElement(COS,"OmegaB").text = "0.0455"
    ET.SubElement(COS,"H0").text = "70.4"
    ET.SubElement(COS,"sigma8").text = "0.810"
    ET.SubElement(COS,"ns").text = "0.967"
    part = ET.SubElement(root,"particles")
    mass = ET.SubElement(part,"mass")
    mass.text = "9.363945573568344e8"
    mass.set("units","Msol/h")
    ET.SubElement(part,"number").text = str(2160**3)
    part = ET.SubElement(root,"boxSize")
    part.text = "500.0"
    part.set("units","Mpc/h")
    zfile = "/Users/amerson/Data/Simulations/ms-w7/redshift_list"
    dtype = [("iz",int),("z",float)]
    zdata = np.loadtxt(zfile,usecols=[0,1],dtype=dtype,skiprows=1).view(np.recarray)
    snaps = ET.SubElement(root,"snapshots")
    for i,z in zip(zdata.iz,zdata.z):
        snap = ET.SubElement(snaps,"snapshot")
        snap.text = str(z)
        snap.set("number",str(i))
    tree = ET.ElementTree(root)
    tree.write("ms-w7.xml")
    formatFile("ms-w7.xml")
