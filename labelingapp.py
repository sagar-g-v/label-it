# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 10:16:54 2019

@author: sagarg9
"""
import os
import sys
import cv2
import copy
import json
import math
import time
import natsort
import hashlib
import platform
import functools
import numpy as np
import pandas as pd
from PIL import Image, ImageQt, ImageEnhance
from qtpy.QtWidgets import (QMainWindow, QFileDialog, QApplication, QWidget, QLabel, QScrollBar, QMenu, QToolButton,
                            QSpinBox, QScrollArea, QSlider, QAction, QPushButton, QGridLayout, QLineEdit, QComboBox,
                            QCheckBox, QCompleter, QDockWidget, QHBoxLayout, QGroupBox, QVBoxLayout,QMessageBox, QColorDialog,
                            QDialogButtonBox,QProgressBar, QDialog, QToolBar, QDesktopWidget, QAbstractSpinBox)
from qtpy.QtCore import (QBasicTimer, Qt, QCoreApplication, QRectF, QRect, QSettings, QSize, QPointF,QPoint,Signal,QTimer,
                         Slot,)
from qtpy.QtGui import (QIcon, QPicture,QPixmap, QColor, QPen,QBrush, QFont, QPainterPath, QFontMetrics, QImage,
                        QCursor, QPainter,QIntValidator,QPalette, QFontDatabase)
from qtpy import QT_VERSION
QT5 = QT_VERSION[0] == '5'

__appname__ = 'mylabelingapp'
__icondir = 'icons'
__appIcon__ = os.path.join(__icondir,'appicon.png')

# CREDIT: https://github.com/tzutalin/labelImg
# TODO: occluded and occluded by
# TODO: integration of map : OpenStreetMap
# TODO: add output formats, parse and covert
# refer (https://github.com/eweill/convert-datasets) or something else
#   - Pascal VOC
#   - YOLO
#   - TFrecords
#   - kitti
#   - COCO

DEFAULT_LINE_COLOR = QColor(0, 200, 0, 255)
DEFAULT_FILL_COLOR = QColor(255, 0, 0, 50)
DEFAULT_SELECT_LINE_COLOR = QColor(255, 255, 255)
DEFAULT_SELECT_FILL_COLOR = QColor(0, 128, 255, 50)
DEFAULT_VERTEX_FILL_COLOR = QColor(255, 255, 255, 255)
DEFAULT_HVERTEX_FILL_COLOR = QColor(255, 0, 0)
DEFAULT_CONTROLPOINT_FILL_COLOR = QColor(255, 255, 255, 50)
DEFAULT_HCONTROLPOINT_FILL_COLOR = QColor(255, 255, 0, 255)

class Shape(object):

    P_SQUARE, P_ROUND = list(range(2))

    MOVE_VERTEX, NEAR_VERTEX  = list(range(2))

    # The following class variables influence the drawing of all shape objects.
    line_color = DEFAULT_LINE_COLOR
    fill_color = DEFAULT_FILL_COLOR
    select_line_color = DEFAULT_SELECT_LINE_COLOR
    select_fill_color = DEFAULT_SELECT_FILL_COLOR
    vertex_fill_color = DEFAULT_VERTEX_FILL_COLOR
    hvertex_fill_color = DEFAULT_HVERTEX_FILL_COLOR
    controlpoint_fill_color = DEFAULT_CONTROLPOINT_FILL_COLOR
    hcontrolpoint_fill_color = DEFAULT_HCONTROLPOINT_FILL_COLOR
    point_type = P_ROUND
    point_size = 6
    scale = 1.0

    def __init__(self, label=None, line_color=None, shape_type=None):
        self.label = label
        self.points = []
        self.controlvalues = []
        self.fill = False
        self.selected = False
        self.shape_type = shape_type

        self.shape_property = {
            "shape_type": self.shape_type, 
            "line_color": None, 
            "points": [],
            "controlvalues": [],
            "fill_color": None, 
            "label": self.label
        }
        self._highlightIndex = None
        self._highlightContolsIndex = None
        self._highlightMode = self.NEAR_VERTEX
        self._highlightSettings = {
            self.MOVE_VERTEX: (1.5, self.P_SQUARE),
            self.NEAR_VERTEX: (4, self.P_ROUND),
        }
        self._closed = False

        if line_color is not None:
            # Override the class line_color attribute
            # with an object attribute. Currently this
            # is used for drawing the pending line a different color.
            self.line_color = line_color

        self.shape_type = shape_type

    @property
    def shape_type(self):
        return self._shape_type

    @shape_type.setter
    def shape_type(self, value):
        if value is None:
            value = 'polygon'
        if value not in ['polygon', 'rectangle', 'point',
           'line', 'circle', 'polyline', 'cube']:
            raise ValueError('Unexpected shape_type: {}'.format(value))
        self._shape_type = value
    
    def updateProperty(self, update = dict()):
        self.shape_property["points"] = [{"x":point.x(),"y":point.y()} \
                                        for point in self.points]
        self.shape_property["controlvalues"] = [{"x":value.x(),"y":value.y()} \
                                        for value in self.controlvalues]
    
    def close(self):
        self._closed = True

    def addPoint(self, point):
        if self.shape_type == 'polygon' and self.points and point == self.points[0]:
            self.close()
        else:
            self.points.append(point)

    def popPoint(self):
        if self.points:
            return self.points.pop()
        return None

    def insertPoint(self, i, point):
        self.points.insert(i, point)

    def removePoint(self, i):
        point = self.points[i]
        self.points.remove(point)

    def isClosed(self):
        return self._closed

    def size(self):
        if self.isClosed:
            rect = self.getRectFromLine(self.points[0], self.points[1])
            if self.shape_type == 'rectangle':
                return [rect.width(), rect.height()]
            if self.shape_type == 'circle' or self.shape_type == 'line':
                return abs(rect.width())
        return None

    def getArea(self):
        if self.shape_type == 'rectangle':
            w,h = self.size()
            return w*h
        elif self.shape_type == 'circle':
            r = self.size()
            return np.pi*(r**2)
        else:
            assert False, 'couldnt find area for this shape'

    def setOpen(self):
        self._closed = False

    def getRectFromLine(self, pt1, pt2):
        x1, y1 = pt1.x(), pt1.y()
        x2, y2 = pt2.x(), pt2.y()
        return QRectF(x1, y1, x2 - x1, y2 - y1)

    def getCubeFromPoints(self, points):
        cubePath = QPainterPath()
        cubepoints = [points[0], points[1], QPoint(points[0].x(),points[1].y()), QPoint(points[1].x(),points[0].y()),
                      points[2],points[3],QPoint(points[2].x(),points[3].y()), QPoint(points[3].x(),points[2].y())]

        frontRect = self.getRectFromLine(points[0],points[1])
        backRect = self.getRectFromLine(points[2],points[3])
        cubePath = QPainterPath()

        cubePath.addRect(frontRect)
        cubePath.addRect(backRect)
        for i in range(4):
            cubePath.moveTo(cubepoints[i])
            cubePath.lineTo(cubepoints[i+4])
        return cubePath

    def paint(self, painter):
        if self.points:
            color = self.select_line_color \
                if self.selected else self.line_color
            penwidth = round(2.0 / self.scale)
            pen = QPen(color, penwidth, Qt.SolidLine, Qt.RoundCap,
                        Qt.RoundJoin)
            painter.setPen(pen)

            line_path = QPainterPath()
            vrtx_path = QPainterPath()
            ctrx_path = QPainterPath()
            if self.shape_type == 'rectangle':
                assert len(self.points) in [1, 2]
                if len(self.points) == 2:
                    rectangle = self.getRectFromLine(*self.points)
                    line_path.addRect(rectangle)
                    for i in range(4):
                        if len(self.controlvalues) < 4:
                            self.controlvalues.append(QPoint())
                        self.drawVertex(vrtx_path, i)
                        self.drawControlPoint(ctrx_path, i)
                else:
                    for i in range(len(self.points)):
                        self.drawVertex(vrtx_path, i)

            elif self.shape_type == "cube":
                assert len(self.points) in [1, 2, 3, 4]
                if len(self.points) == 2:
                    rectangle = self.getRectFromLine(*self.points)
                    line_path.addRect(rectangle)
                elif len(self.points) == 4:
                    cube = self.getCubeFromPoints(self.points)
                    line_path.addPath(cube)
                for i in range(len(self.points)):
                    self.drawVertex(vrtx_path, i)

            elif self.shape_type == "circle":
                assert len(self.points) in [1, 2]
                if len(self.points) == 2:
                    rectangle = self.getCircleRectFromLine(self.points)
                    line_path.addEllipse(rectangle)
                for i in range(len(self.points)):
                    self.drawVertex(vrtx_path, i)

            elif self.shape_type == "polyline":
                line_path.moveTo(self.points[0])
                for i, p in enumerate(self.points):
                    line_path.lineTo(p)
                    self.drawVertex(vrtx_path, i)
            else:
                line_path.moveTo(self.points[0])
                # Uncommenting the following line will draw 2 paths
                # for the 1st vertex, and make it non-filled, which
                # may be desirable.
                # self.drawVertex(vrtx_path, 0)

                for i, p in enumerate(self.points):
                    line_path.lineTo(p)
                    self.drawVertex(vrtx_path, i)
                if self.isClosed():
                    line_path.lineTo(self.points[0])

            pen.setColor(color)
            painter.setPen(pen)
            painter.drawPath(line_path)
            painter.drawPath(vrtx_path)
            pen.setColor(QColor(200,200,0))
            painter.setPen(pen)
            painter.drawPath(ctrx_path)

            painter.fillPath(vrtx_path, self.vertex_fill_color)
            painter.fillPath(ctrx_path, self.controlpoint_fill_color)
            if self.fill:
                color = self.select_fill_color \
                    if self.selected else self.fill_color
                painter.fillPath(line_path, color)

    def drawVertex(self, path, i):
        d = self.point_size/(self.scale*0.7) if self.scale > 1.4 else self.point_size
        shape = self.point_type
        if i == self._highlightIndex:
            size, shape = self._highlightSettings[self._highlightMode]
            d *= size
        if self._highlightIndex is not None:
            self.vertex_fill_color = self.hvertex_fill_color
        else:
            self.vertex_fill_color = Shape.vertex_fill_color


        if self.shape_type == 'rectangle':
            signleft, signright = [1,-1,1,-1], [1,-1,-1,1]
            if i <= 1:
                point = self.points[i]
            else:
                point = QPoint(self.points[i-2].x(),self.points[abs(i-3)].y())

            loffset, roffset = signleft[i]*d / 3.0, signright[i]*d / 3.0
            edgepath = QPainterPath()
            edgepath.moveTo(QPoint(point.x() + loffset,point.y()))
            edgepath.lineTo(point)
            edgepath.lineTo(QPoint(point.x(),point.y() + roffset))

            if shape == self.P_SQUARE:
                path.addRect(point.x() - d / 3.0, point.y() - d / 3.0, 0.66*d, 0.66*d)
            elif shape == self.P_ROUND:
                #path.addEllipse(point, d / 3.0, d / 3.0)
                #path.addPath(edgepath)
                pass
            else:
                assert False, "unsupported vertex shape"
        else:
            point = self.points[i]
            if shape == self.P_SQUARE:
                path.addRect(point.x() - d / 3.0, point.y() - d / 3.0, 0.66*d, 0.66*d)
            elif shape == self.P_ROUND:
                path.addEllipse(point, d / 3.0, d / 3.0)
            else:
                assert False, "unsupported vertex shape"

    def drawLabel(self, path, i):
        pass

    def drawControlPoint(self, path, i):
        d = self.point_size/(self.scale*0.7) if self.scale > 1.4 else self.point_size
        shape = self.point_type
        if i == self._highlightContolsIndex:
            size, shape = self._highlightSettings[self._highlightMode]
            d *= size
        if self._highlightContolsIndex is not None:
            self.controlpoint_fill_color = self.hcontrolpoint_fill_color
        else:
            self.controlpoint_fill_color = Shape.controlpoint_fill_color

        point = self.getControlPoint(i)
        if shape == self.P_SQUARE:
            path.moveTo(QPointF(point.x() - d / 4.0, point.y()))
            path.lineTo(QPointF(point.x(), point.y() - d / 4.0))
            path.lineTo(QPointF(point.x()+ d / 4.0, point.y()))
            path.lineTo(QPointF(point.x(), point.y() + d / 4.0))
            path.lineTo(QPointF(point.x() - d / 4.0, point.y()))
        elif shape == self.P_ROUND:
            pass
            #path.addEllipse(point, d / 4.0, d / 4.0)
        else:
            assert False, "unsupported vertex shape"

    def drawExtendedShape(self, path, i):

        pass

    def getControlPoint(self, i):
        if self.shape_type == 'rectangle':
            value = self.controlvalues[i]
            points = [self.points[0],self.points[1],
                           QPoint(self.points[0].x(),self.points[1].y()),
                           QPoint(self.points[1].x(),self.points[0].y())]
            if value and value.isNull():
                value = QPoint()
            if (i == 0 or i == 1):
                cpoint = points[i] + QPoint((points[abs(i-3)].x()-points[i].x())/2,value.y())
            elif (i == 2 or i == 3):
                cpoint = points[i] + QPoint(value.x(), (points[i-2].y()-points[i].y())/2)
            return cpoint

    def nearestVertex(self, point, epsilon):
        min_distance = float('inf')
        nearest = None
        if self.shape_type == 'rectangle':
            points = [self.points[0],self.points[1],
                           QPoint(self.points[0].x(),self.points[1].y()),
                           QPoint(self.points[1].x(),self.points[0].y())]
        else:
            points = self.points

        for i, p in enumerate(points):
            dist = distancetopoint(p, point)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                nearest = i
        return nearest

    def nearestEdge(self, point, epsilon):
        min_distance = float('inf')
        nearest = None
        offset = epsilon//2 + 2.0
        if self.shape_type == 'rectangle':
            points = [self.points[0],self.points[1],
                           QPoint(self.points[0].x(),self.points[1].y()),
                           QPoint(self.points[1].x(),self.points[0].y())]
        else:
            points = self.points

        for i in range(len(points)):
            if self.shape_type == 'rectangle':
                if i == 0:
                    line = [points[0]+QPoint(offset,0.0), points[3]-QPoint(offset,0.0)]
                elif i == 1:
                    line = [points[1]-QPoint(offset,0.0), points[2]+QPoint(offset,0.0)]
                elif i == 2:
                    line = [points[2]-QPoint(0.0,offset), points[0]+QPoint(0.0,offset)]
                else:
                    line = [points[3]+QPoint(0.0,offset), points[1]-QPoint(0.0,offset)]
            else:
                line = [points[i - 1], points[i]]

            dist = distancetoline(point, line)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                nearest = i
        return nearest

    def nearestControlPoint(self, point, epsilon):
        min_distance = float('inf')
        nearest = None

        if self.shape_type == 'rectangle':
            for i in range(4):
                cpoint = self.getControlPoint(i)
                dist = distancetopoint(cpoint, point)
                if dist <= epsilon and dist < min_distance:
                    min_distance = dist
                    nearest = i
            return nearest

    def containsPoint(self, point):
        return self.makePath().contains(point)

    def getCircleRectFromLine(self, line):
        """Computes parameters to draw with `QPainterPath::addEllipse`"""
        if len(line) != 2:
            return None
        (c, point) = line
        r = line[0] - line[1]
        d = math.sqrt(math.pow(r.x(), 2) + math.pow(r.y(), 2))
        rectangle = QRectF(c.x() - d, c.y() - d, 2 * d, 2 * d)
        return rectangle

    def makePath(self):
        if self.shape_type == 'rectangle':
            path = QPainterPath()
            if len(self.points) == 2:
                rectangle = self.getRectFromLine(*self.points)
                path.addRect(rectangle)
        elif self.shape_type == "circle":
            path = QPainterPath()
            if len(self.points) == 2:
                rectangle = self.getCircleRectFromLine(self.points)
                path.addEllipse(rectangle)
        else:
            path = QPainterPath(self.points[0])
            for p in self.points[1:]:
                path.lineTo(p)
        return path

    def boundingRect(self):
        return self.makePath().boundingRect()

    def moveBy(self, offset):
        self.points = [p + offset for p in self.points]

    def moveVertexBy(self, i, offset):
        self.points[i] = self.points[i] + offset

    def highlightVertex(self, i, action):
        self._highlightIndex = i
        self._highlightMode = action

    def hightlightControlPoint(self, i, action):
        self._highlightContolsIndex = i
        self._highlightMode = action

    def highlightClear(self):
        self._highlightIndex = None
        self._highlightContolsIndex = None

    def copy(self):
        shape = Shape(label=self.label, shape_type=self.shape_type)
        shape.points = [copy.deepcopy(p) for p in self.points]
        shape.fill = self.fill
        shape.selected = self.selected
        shape._closed = self._closed
        shape.line_color = copy.deepcopy(self.line_color)
        shape.fill_color = copy.deepcopy(self.fill_color)
        return shape

    def __len__(self):
        return len(self.points)

    def __getitem__(self, key):
        return self.points[key]

    def __setitem__(self, key, value):
        self.points[key] = value

# TODO(unknown):
# - [maybe] Find optimal epsilon value.

CURSOR_DEFAULT = Qt.ArrowCursor
CURSOR_POINT = Qt.PointingHandCursor
CURSOR_DRAW = Qt.CrossCursor
CURSOR_MOVE = Qt.ClosedHandCursor
CURSOR_GRAB = Qt.OpenHandCursor
CURSOR_VSIZE = Qt.SizeVerCursor
CURSOR_HSIZE = Qt.SizeHorCursor

class Canvas(QWidget):

    zoomRequest = Signal(int, QPoint)
    scrollRequest = Signal(int, int)
    newShape = Signal()
    selectionChanged = Signal(bool)
    shapeMoved = Signal()
    drawingPolygon = Signal(bool)
    edgeSelected = Signal(bool)

    CREATE, EDIT = 0, 1

    # polygon, rectangle, line, or point
    _createMode = 'polygon'

    _fill_drawing = False

    def __init__(self, *args, **kwargs):
        self.epsilon = kwargs.pop('epsilon', 10.0)
        super(Canvas, self).__init__(*args, **kwargs)
        # Initialise local state.
        self.mode = self.EDIT
        self.shapes = []
        self.shapesBackups = []
        self.current = None
        self.selectedShape = None  # save the selected shape here
        self.autoselect = -1
        self.selectedShapeCopy = None
        self.lineColor = QColor(0, 0, 255)
        # self.line represents:
        #   - createMode == 'polygon': edge from last point to current
        #   - createMode == 'rectangle': diagonal line of the rectangle
        #   - createMode == 'line': the line
        #   - createMode == 'point': the point
        self.line = Shape(line_color=self.lineColor)
        self.prevPoint = QPoint()
        self.prevMovePoint = QPoint()
        self.offsets = QPoint(), QPoint()
        self.scale = 1.0
        self.pixmap = QPixmap()
        # TODO: hiding and locking shapes need to be added
        self.visible = {}
        self.locked = {}
        self._hideBackround = False
        self.hideBackround = False
        self.hShape = None
        self.hVertex = None
        self.hEdge = None
        self.hCpoint = None
        self.movingShape = False
        self.smoothimage = False
        self._painter = QPainter()
        self._cursor = CURSOR_DEFAULT
        # Menus:
        self.menus = (QMenu(), QMenu())
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.WheelFocus)

    def fillDrawing(self):
        return self._fill_drawing

    def setFillDrawing(self, value):
        self._fill_drawing = value

    @property
    def createMode(self):
        return self._createMode

    @createMode.setter
    def createMode(self, value):
        if value not in ['polygon', 'rectangle', 'circle',
           'line', 'point', 'polyline', 'cube']:
            raise ValueError('Unsupported createMode: %s' % value)
        self._createMode = value

    def storeShapes(self):
        shapesBackup = []
        for shape in self.shapes:
            shapesBackup.append(shape.copy())
        if len(self.shapesBackups) >= 10:
            self.shapesBackups = self.shapesBackups[-9:]
        self.shapesBackups.append(shapesBackup)

    @property
    def isShapeRestorable(self):
        if len(self.shapesBackups) < 2:
            return False
        return True

    def restoreShape(self):
        if not self.isShapeRestorable:
            return
        self.shapesBackups.pop()  # latest
        shapesBackup = self.shapesBackups.pop()
        self.shapes = shapesBackup
        self.storeShapes()
        self.repaint()

    def enterEvent(self, ev):
        self.overrideCursor(self._cursor)

    def leaveEvent(self, ev):
        self.restoreCursor()

    def focusOutEvent(self, ev):
        self.restoreCursor()

    def isVisible(self, shape):
        return self.visible.get(shape, True)

    def drawing(self):
        return self.mode == self.CREATE

    def editing(self):
        return self.mode == self.EDIT

    def setEditing(self, value=True):
        self.mode = self.EDIT if value else self.CREATE
        if not value:  # Create
            self.unHighlight()
            self.deSelectShape()

    def unHighlight(self):
        if self.hShape:
            self.hShape.highlightClear()
        self.hVertex = self.hShape = None

    def selectedVertex(self):
        return self.hVertex is not None

    def selectedEdge(self):
        return self.hEdge is not None

    def selectedControlPoint(self):
        return self.hCpoint is not None

    def mouseMoveEvent(self, ev):
        """Update line with last point and current coordinates."""
        try:
            if QT5:
                pos = self.transformPos(ev.pos())
            else:
                pos = self.transformPos(ev.posF())
        except AttributeError:
            return
        if not self.outOfPixmap(pos):
            self.setToolTip("Image")
            self.setStatusTip('')
            self.parent().window().labelCoordinates.setText(
            'X: <b>%d</b>; Y: <b>%d</b> ' % (pos.x(), pos.y()))
        else:
            self.setToolTip("Canvas")
            self.setStatusTip('')
            self.parent().window().labelCoordinates.clear()

        self.prevMovePoint = pos
#        self.restoreCursor()
        self.overrideCursor(Qt.ArrowCursor)

        # Polygon drawing.
        if self.drawing():
            self.line.shape_type = self.createMode

            self.overrideCursor(CURSOR_DRAW)
            if not self.current:
                return

            color = self.lineColor
            if self.outOfPixmap(pos):
                # Don't allow the user to draw outside the pixmap.
                # Project the point to the pixmap's edges.
                pos = self.intersectionPoint(self.current[-1], pos)
            elif len(self.current) > 1 and self.createMode == 'polygon' and\
                    self.closeEnough(pos, self.current[0]):
                # Attract line to starting point and
                # colorise to alert the user.
                pos = self.current[0]
                color = self.current.line_color
                self.overrideCursor(CURSOR_POINT)
                self.current.highlightVertex(0, Shape.NEAR_VERTEX)

            if self.createMode in ['polygon', 'polyline']:
                self.line[0] = self.current[-1]
                self.line[1] = pos
            elif self.createMode == 'rectangle':
                self.line.points = [self.current[0], pos]
                self.line.close()
            elif self.createMode == 'cube':
                if len(self.current.points) == 1:
                    self.line.points = [self.current[0], pos]
                    self.line.close()
                elif len(self.current.points) < 4:
                    size = self.current.points[1] - self.current.points[0]
                    self.line.points = [QPoint(pos.x()-(size.x()/2),pos.y()-(size.y()/2)),\
                                        QPoint(pos.x()+(size.x()/2),pos.y()+(size.y()/2))]
                    self.line.close()
            elif self.createMode == 'circle':
                self.line.points = [self.current[0], pos]
                self.line.shape_type = "circle"
            elif self.createMode == 'line':
                self.line.points = [self.current[0], pos]
                self.line.close()
            elif self.createMode == 'point':
                self.line.points = [self.current[0]]
                self.line.close()
            self.line.line_color = color
            self.repaint()
            self.current.highlightClear()
            return

        # Polygon copy moving.
        if Qt.RightButton & ev.buttons():
            if self.selectedShapeCopy and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShape(self.selectedShapeCopy, pos)
                self.repaint()
            elif self.selectedShape:
                self.selectedShapeCopy = self.selectedShape.copy()
                self.repaint()
            return

        # Polygon/Vertex moving.
        self.movingShape = False
        if Qt.LeftButton & ev.buttons():
            if self.selectedVertex():
                self.boundedMoveVertex(pos)
                self.repaint()
                self.movingShape = True
            elif self.selectedEdge():
                self.boundedMoveEdge(pos)
                self.repaint()
                self.movingShape = True
            elif self.selectedShape and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShape(self.selectedShape, pos)
                self.repaint()
                self.movingShape = True
            return

        # Just hovering over the canvas, 2 posibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index, eindex, cindex = None, None, None
            index = shape.nearestVertex(pos, self.epsilon)
            eindex = shape.nearestEdge(pos, self.epsilon)
            cindex = shape.nearestControlPoint(pos, 2*self.epsilon)

            if index is not None:
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.hEdge, self.hShape, self.hVertex, self.hCpoint = None, shape, index, None
                shape.highlightVertex(index, shape.MOVE_VERTEX)
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip("Click & drag to move vertex")
                self.update()
                break

            elif eindex is not None:
                if self.selectedEdge():
                    self.hShape.highlightClear()
                if shape.shape_type == 'rectangle':
                    if eindex == 0 or eindex == 1:
                        self.overrideCursor(CURSOR_VSIZE)
                    else:
                        self.overrideCursor(CURSOR_HSIZE)
                self.hEdge, self.hShape, self.hVertex, self.hCpoint = eindex, shape, None, None
                self.setToolTip("Click & drag to move edge")
                self.update()
                break

            elif cindex is not None:
                if self.selectedControlPoint():
                    self.hShape.highlightClear()
                self.hEdge, self.hShape, self.hVertex, self.hCpoint = None, shape, None, cindex
                shape.hightlightControlPoint(cindex, shape.MOVE_VERTEX)
                self.setToolTip("Click & drag to move point")
                self.update()
                break


#            elif shape.containsPoint(pos):# highlight when hovering over shape
#                if self.selectedVertex():
#                    self.hShape.highlightClear()
#                self.hVertex = None
#                self.hShape = shape
#                self.hEdge = None
#                self.setToolTip(
#                    "Click & drag to move shape '%s'" % shape.label)
#                self.overrideCursor(CURSOR_GRAB)
#                self.update()
#                break
        else:  # Nothing found, clear highlights, reset state.
            if self.hShape:
                self.hShape.highlightClear()
                self.update()
            self.hVertex, self.hShape, self.hEdge, self.hCpoint = None, None, None, None
        self.edgeSelected.emit(self.hEdge is not None)

    def addPointToEdge(self):
        if (self.hShape is None and
                self.hEdge is None and
                self.prevMovePoint is None):
            return
        if not self.hShape.shape_type in ['polygon', 'polyline']:
            return
        shape = self.hShape
        index = self.hEdge
        point = self.prevMovePoint
        shape.insertPoint(index, point)
        shape.highlightVertex(index, shape.MOVE_VERTEX)
        self.hShape = shape
        self.hVertex = index
        self.hEdge = None

    def mousePressEvent(self, ev):
        if QT5:
            pos = self.transformPos(ev.pos())
        else:
            pos = self.transformPos(ev.posF())
        if ev.button() == Qt.LeftButton:
            if self.drawing():
                self.overrideCursor(CURSOR_DRAW)
                if self.current:
                    # Add point to existing shape.
                    if self.createMode in ['polygon']:
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if self.current.isClosed():
                            self.finalise()
                    elif self.createMode in ['rectangle', 'circle', 'line']:
                        assert len(self.current.points) == 1
                        self.current.points = self.line.points
                        self.finalise()
                    elif self.createMode == 'polyline':
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if int(ev.modifiers()) == Qt.ControlModifier:
                            self.finalise()
                    elif self.createMode == 'cube':
                        if len(self.current.points) == 1:
                            self.current.points = self.line.points
                        elif len(self.current.points) == 2:
                            self.current.points[2:] = self.line.points
                            self.finalise()

                elif not self.outOfPixmap(pos):
                    # Create new shape.
                    self.drawingPolygon.emit(True)
                    self.current = Shape(shape_type=self.createMode)
                    self.current.addPoint(pos)
                    if self.createMode == 'point':
                        self.finalise()
                    else:
                        if self.createMode == 'circle':
                            self.current.shape_type = 'circle'
                        self.line.points = [pos, pos]
                        self.setHiding()
                        self.update()
            else:
#                self.overrideCursor(Qt.ArrowCursor)
                self.selectShapePoint(pos)
                self.prevPoint = pos
                if self.selectedEdge() and not self.selectedVertex():
                    if int(ev.modifiers()) == Qt.ControlModifier:
                        self.addPointToEdge()
                self.repaint()
        elif ev.button() == Qt.RightButton and self.editing():
            self.selectShapePoint(pos)
            self.prevPoint = pos
            self.repaint()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.RightButton:
#            menu = self.menus[bool(self.selectedShapeCopy)]
            self.restoreCursor()
#            if not menu.exec_(self.mapToGlobal(ev.pos()))\
#               and self.selectedShapeCopy:
#                # Cancel the move by deleting the shadow copy.
#                self.selectedShapeCopy = None
#                self.repaint()
        elif ev.button() == Qt.LeftButton and self.selectedShape:
            self.overrideCursor(CURSOR_GRAB)
        if self.movingShape:
            self.storeShapes()
            self.shapeMoved.emit()

    def endMove(self, copy=False):
        assert self.selectedShape and self.selectedShapeCopy
        shape = self.selectedShapeCopy
        # del shape.fill_color
        # del shape.line_color
        if copy:
            self.shapes.append(shape)
            self.selectedShape.selected = False
            self.selectedShape = shape
            self.repaint()
        else:
            shape.label = self.selectedShape.label
            self.deleteSelected()
            self.shapes.append(shape)
        self.storeShapes()
        self.selectedShapeCopy = None

    def hideBackroundShapes(self, value):
        self.hideBackround = value
        if self.selectedShape:
            # Only hide other shapes if there is a current selection.
            # Otherwise the user will not be able to select a shape.
            self.setHiding(True)
            self.repaint()

    def setHiding(self, enable=True):
        self._hideBackround = self.hideBackround if enable else False

    def canCloseShape(self):
        return self.drawing() and self.current and len(self.current) > 2

    def mouseDoubleClickEvent(self, ev):
        # We need at least 4 points here, since the mousePress handler
        # adds an extra one before this handler is called.
        if self.canCloseShape() and len(self.current) > 3:
            self.current.popPoint()
            self.finalise()

    def selectShape(self, shape):
        self.deSelectShape()
        shape.selected = True
        self.selectedShape = shape
        self.setHiding()
        self.selectionChanged.emit(True)
        self.update()

    def selectShapePoint(self, point):
        """Select the first shape created which contains this point."""
        self.deSelectShape()
        if self.selectedVertex():  # A vertex is marked for selection.
            index, shape = self.hVertex, self.hShape
            shape.highlightVertex(index, shape.MOVE_VERTEX)
            self.selectShape(shape)
            return
        elif self.selectedEdge():
            index, shape = self.hEdge, self.hShape
           # shape.highlightVertex(index, shape.MOVE_EDGE)
            self.selectShape(shape)
            return

        for shape in reversed(self.shapes):
            if self.isVisible(shape) and shape.containsPoint(point):
                self.calculateOffsets(shape,point)
                self.selectShape(shape)
                return

    def calculateOffsets(self, shape, point):
        rect = shape.boundingRect()
        x1 = rect.x() - point.x()
        y1 = rect.y() - point.y()
        x2 = (rect.x() + rect.width() - 1) - point.x()
        y2 = (rect.y() + rect.height() - 1) - point.y()
        self.offsets = QPoint(x1, y1), QPoint(x2, y2)

    def boundedMoveVertex(self, pos):
        index, shape = self.hVertex, self.hShape
        if shape.shape_type == 'rectangle':
            w, h = shape.size()
            rectanglePoints = [shape[0], shape[1], QPoint(shape[0].x(),shape[1].y()), QPoint(shape[1].x(),shape[0].y())]
            point = rectanglePoints[index]
            if self.outOfPixmap(pos):
                pos = self.intersectionPoint(point, pos)
            shiftPos = pos - point
            if index == 0:
                if h - shiftPos.y() >= 10.0 :
                    shape.moveVertexBy(index, QPointF(0.0, shiftPos.y()))
                if w - shiftPos.x() >= 10.0 :
                    shape.moveVertexBy(index, QPointF(shiftPos.x(), 0.0))
            elif index == 1:
                if h + shiftPos.y() >= 10.0 :
                    shape.moveVertexBy(index, QPointF(0.0, shiftPos.y()))
                if w + shiftPos.x() >= 10.0 :
                    shape.moveVertexBy(index, QPointF(shiftPos.x(),0.0))
            elif index == 2:
                shiftPos = pos - shape[0]
                if w - shiftPos.x() >= 10.0:
                    shape.moveVertexBy(0, QPointF(shiftPos.x(), 0.0))
                shiftPos = pos - shape[1]
                if h + shiftPos.y() >= 10.0:
                    shape.moveVertexBy(1, QPointF(0.0, shiftPos.y()))
            else:
                shiftPos = pos - shape[1]
                if w + shiftPos.x() >= 10.0:
                    shape.moveVertexBy(1, QPointF(shiftPos.x(), 0.0))
                shiftPos = pos - shape[0]
                if h - shiftPos.y() >= 10.0:
                    shape.moveVertexBy(0, QPointF(0.0, shiftPos.y()))
        else:
            point = shape[index]
            if self.outOfPixmap(pos):
                pos = self.intersectionPoint(point, pos)
            shape.moveVertexBy(index, pos - point)

    def boundedMoveEdge(self, pos):
        index, shape = self.hEdge, self.hShape
        if shape.shape_type == 'rectangle':
            rectanglePoints = [shape[0], shape[1], QPoint(shape[0].x(),shape[1].x()), QPoint(shape[1].x(),shape[0].x())]
            w, h = shape.size()
            point = rectanglePoints[index]
            if self.outOfPixmap(pos):
                #ERROR:self.canvas.scale when trying to move edge out of image
                pos = self.intersectionPoint(point, pos)

            shiftPos = pos - point
            shift = None
            if index == 1:
                if h + shiftPos.y() >= 10.0:
                   shift = QPointF(0.0, shiftPos.y())
                   shape.moveVertexBy(1, shift)
            elif index == 0:
                if h - shiftPos.y() >= 10.0:
                   shift = QPointF(0.0, shiftPos.y())
                   shape.moveVertexBy(0, shift)
            elif index == 2:
                if w - shiftPos.x() >= 10.0:
                    shift = QPointF(shiftPos.x(), 0.0)
                    shape.moveVertexBy(0, shift)
            elif index == 3:
                if w + shiftPos.x() >= 10.0:
                    shift = QPointF(shiftPos.x(), 0.0)
                    shape.moveVertexBy(1, shift)

    def boundedMoveShape(self, shape, pos):
        if self.outOfPixmap(pos):
            return False  # No need to move
        o1 = pos + self.offsets[0]
        if self.outOfPixmap(o1):
            pos -= QPoint(min(0, o1.x()), min(0, o1.y()))
        o2 = pos + self.offsets[1]
        if self.outOfPixmap(o2):
            pos += QPoint(min(0, self.pixmap.width() - o2.x()),
                                 min(0, self.pixmap.height() - o2.y()))
        # XXX: The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason.
        # self.calculateOffsets(self.selectedShape, pos)
        dp = pos - self.prevPoint
        if dp:
            shape.moveBy(dp)
            self.prevPoint = pos
            return True
        return False

    def deSelectShape(self):
        if self.selectedShape:
            self.selectedShape.selected = False
            self.selectedShape = None
            self.setHiding(False)
            self.selectionChanged.emit(False)
            self.update()

    def deleteSelected(self):
        if self.selectedShape:
            shape = self.selectedShape
            self.shapes.remove(self.selectedShape)
            self.storeShapes()
            self.selectedShape = None
            self.update()
            return shape

    def copySelectedShape(self):
        if self.selectedShape:
            shape = self.selectedShape.copy()
            self.deSelectShape()
            self.shapes.append(shape)
            self.storeShapes()
            shape.selected = True
            self.selectedShape = shape
            self.boundedShiftShape(shape)
            return shape

    def boundedShiftShape(self, shape):
        # Try to move in one direction, and if it fails in another.
        # Give up if both fail.
        point = shape[0]
        offset = QPoint(2.0, 2.0)
        self.calculateOffsets(shape, point)
        self.prevPoint = point
        if not self.boundedMoveShape(shape, point - offset):
            self.boundedMoveShape(shape, point + offset)

    def paintEvent(self, event):
        if not self.pixmap:
            return super(Canvas, self).paintEvent(event)
        p = self._painter
        p.begin(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.HighQualityAntialiasing)
        if self.smoothimage:
            p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.scale(self.scale, self.scale)
        p.translate(self.offsetToCenter())

        p.drawPixmap(0, 0, self.pixmap)
        Shape.scale = self.scale
        for shape in self.shapes:
            if (shape.selected or not self._hideBackround) and \
                    self.isVisible(shape):
                shape.fill = shape.selected or shape == self.hShape
                shape.paint(p)
        if self.current:
            self.current.paint(p)
            self.line.paint(p)
        if self.selectedShapeCopy:
            self.selectedShapeCopy.paint(p)

        if (self.fillDrawing() and self.createMode == 'polygon' and
                self.current is not None and len(self.current.points) >= 2):
            drawing_shape = self.current.copy()
            drawing_shape.addPoint(self.line[1])
            drawing_shape.fill = True
            drawing_shape.fill_color.setAlpha(64)
            drawing_shape.paint(p)

        if self.hShape and self.hEdge and self.hShape.shape_type in ['polygon', 'polyline']:
            dotpath = QPainterPath()
            d = Shape.point_size/(self.scale*0.7) if self.scale > 1.4 else Shape.point_size
            dotpath.addEllipse(self.prevMovePoint, d / 2.0, d / 2.0)
            p.drawPath(dotpath)
            p.fillPath(dotpath, Shape.vertex_fill_color)

        self.setAutoFillBackground(True)

        if not self.pixmap.isNull():
            pal = QPalette()
            pal.setColor(self.backgroundRole(), QColor(50, 50, 50, 255))
            self.setPalette(pal)
            self.setEnabled(True)
        else:
            pal = QPalette()
            pal.setColor(self.backgroundRole(), QColor(200, 200, 200, 255))
            self.drawlogo(p)
            self.setPalette(pal)
            self.setEnabled(False)
        p.end()

    def drawlogo(self, p):
        icon = QIcon(__appIcon__)
        defaultPixmap = icon.pixmap(QSize(100,100),QIcon.Disabled)
        p.drawPixmap(-1*defaultPixmap.width()/2,-1*defaultPixmap.height()/2,defaultPixmap)
        # and draw text
        p.setFont(QFont('Consolas', 20,10))
        p.setPen(QPen(QColor(140,140,140)))
        p.drawText((-1*defaultPixmap.width()/2)-10,(defaultPixmap.height()/2)+20,'empty :(')

    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(Canvas, self).size()
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QPoint(x, y)

    def outOfPixmap(self, p):
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() < w and 0 <= p.y() < h)

    def finalise(self):
        assert self.current
        self.current.close()
#        if self.current.shape_type == 'polyline':
#            print(self.current.points)
        if self.current.shape_type == 'rectangle':
            w,h =  self.current.size()
            points = self.current.points

            if w < 0 and h < 0:
                self.current.points[0] = QPoint(points[1].x(),points[1].y())
                self.current.points[1] = QPoint(points[0].x()-w,points[0].y()-h)
            elif w < 0:
                self.current.points[0] = QPoint(points[1].x(),points[0].y())
                self.current.points[1] = QPoint(points[0].x()-w,points[1].y())
            elif h < 0:
                self.current.points[0] = QPoint(points[0].x(),points[1].y())
                self.current.points[1] = QPoint(points[1].x(),points[0].y()-h)

            w,h =  self.current.size()
            if not(w < 10 or h < 10):
                self.shapes.append(self.current)
                self.storeShapes()
        else:
            self.shapes.append(self.current)
            self.storeShapes()
        self.current = None
        self.setHiding(False)
        self.newShape.emit()
        self.drawingPolygon.emit(False)
        self.update()

    def closeEnough(self, p1, p2):
        # d = distance(p1 - p2)
        # m = (p1-p2).manhattanLength()
        # print "d %.2f, m %d, %.2f" % (d, m, d - m)
        return distance(p1 - p2) < self.epsilon

    def intersectionPoint(self, p1, p2):
        # Cycle through each image edge in clockwise fashion,
        # and find the one intersecting the current line segment.
        # http://paulbourke.net/geometry/lineline2d/
        size = self.pixmap.size()
        points = [(0, 0),
                  (size.width() - 1, 0),
                  (size.width() - 1, size.height() - 1),
                  (0, size.height() - 1)]
        x1, y1 = p1.x(), p1.y()
        x2, y2 = p2.x(), p2.y()
        d, i, (x, y) = min(self.intersectingEdges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        if (x, y) == (x1, y1):
            # Handle cases where previous point is on one of the edges.
            if x3 == x4:
                return QPoint(x3, min(max(0, y2), max(y3, y4)))
            else:  # y3 == y4
                return QPoint(min(max(0, x2), max(x3, x4)), y3)
        return QPoint(x, y)

    def intersectingEdges(self, point1, point2, points):
        """Find intersecting edges.

        For each edge formed by `points', yield the intersection
        with the line segment `(x1,y1) - (x2,y2)`, if it exists.
        Also return the distance of `(x2,y2)' to the middle of the
        edge along with its index, so that the one closest can be chosen.
        """
        (x1, y1) = point1
        (x2, y2) = point2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                # This covers two cases:
                #   nua == nub == 0: Coincident
                #   otherwise: Parallel
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QPoint((x3 + x4) / 2, (y3 + y4) / 2)
                d = distance(m - QPoint(x2, y2))
                yield d, i, (x, y)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    def wheelEvent(self, ev):
        if QT5:
            mods = ev.modifiers()
            delta = ev.angleDelta()
            #print(delta)
            if Qt.ControlModifier == int(mods):
                # with Ctrl/Command key
                # zoom
                self.zoomRequest.emit(delta.y(), ev.pos())
            else:
                # scroll
                self.scrollRequest.emit(delta.x(), Qt.Horizontal)
                self.scrollRequest.emit(delta.y(), Qt.Vertical)
        else:
            if ev.orientation() == Qt.Vertical:
                mods = ev.modifiers()
                if Qt.ControlModifier == int(mods):
                    # with Ctrl/Command key
                    self.zoomRequest.emit(ev.delta(), ev.pos())
                else:
                    self.scrollRequest.emit(
                        ev.delta(),
                        Qt.Horizontal
                        if (Qt.ShiftModifier == int(mods))
                        else Qt.Vertical)
            else:
                self.scrollRequest.emit(ev.delta(), Qt.Horizontal)
        ev.accept()

    def keyPressEvent(self, ev):
        key = ev.key()
        if key == Qt.Key_Escape and self.current:
            self.current = None
            self.drawingPolygon.emit(False)
            self.update()
        elif key == Qt.Key_Return and self.canCloseShape():
            self.finalise()

        mods = ev.modifiers()
        if Qt.ShiftModifier == mods:
            #print("Shift pressed")
            if key == Qt.Key_Right and self.shapes:
                if self.selectedShape:
                    self.autoselect = self.shapes.index(self.selectedShape)
                self.deSelectShape()
                self.autoselect += 1
                if self.autoselect >= len(self.shapes):
                    self.autoselect = -1
                shape = self.shapes[self.autoselect] if not self.autoselect == -1 else self.shapes[0]
                self.selectShape(shape)
            elif key == Qt.Key_Left and self.shapes:
                if self.selectedShape:
                    self.autoselect = self.shapes.index(self.selectedShape)
                self.deSelectShape()
                self.autoselect -= 1
                if self.autoselect < 0:
                    self.autoselect = len(self.shapes)-1
                shape = self.shapes[self.autoselect]
                self.selectShape(shape)

        elif Qt.ControlModifier == mods:
            #print("Ctrl pressed")
            if key == Qt.Key_Left and self.selectedShape:
                self.moveOnePixel('leftEOut')
            elif key == Qt.Key_Right and self.selectedShape:
                self.moveOnePixel('rightEOut')
            elif key == Qt.Key_Up and self.selectedShape:
                self.moveOnePixel('upEOut')
            elif key == Qt.Key_Down and self.selectedShape:
                self.moveOnePixel('downEOut')
            if key == Qt.Key_T:
                self.smoothimage = not self.smoothimage
                self.update()
        elif (Qt.ControlModifier|Qt.ShiftModifier) == mods:
            #print("Ctrl+Shift pressed")
            if key == Qt.Key_Left and self.selectedShape:
                self.moveOnePixel('leftEIn')
            elif key == Qt.Key_Right and self.selectedShape:
                self.moveOnePixel('rightEIn')
            elif key == Qt.Key_Up and self.selectedShape:
                self.moveOnePixel('upEIn')
            elif key == Qt.Key_Down and self.selectedShape:
                self.moveOnePixel('downEIn')
        elif Qt.AltModifier == mods:
            #print("Alt pressed")
            if key == Qt.Key_Up and self.selectedShape:
                self.moveOnePixel('expandOut')
            elif key == Qt.Key_Down and self.selectedShape:
                self.moveOnePixel('expandIn')
        else:
            if key == Qt.Key_Left and self.selectedShape:
                self.moveOnePixel('left')
            elif key == Qt.Key_Right and self.selectedShape:
                self.moveOnePixel('right')
            elif key == Qt.Key_Up and self.selectedShape:
                self.moveOnePixel('up')
            elif key == Qt.Key_Down and self.selectedShape:
                self.moveOnePixel('down')

    def moveOutOfBound(self, step):
        points = [p1+p2 for p1, p2 in zip(self.selectedShape.points, [step]*len(self.selectedShape.points))]
        return True in map(self.outOfPixmap, points)

    def moveOnePixel(self, direction):
        if direction == 'left' and not self.moveOutOfBound(QPointF(-1.0, 0)):
            self.selectedShape.moveBy(QPointF(-1.0, 0))
        elif direction == 'right' and not self.moveOutOfBound(QPointF(1.0, 0)):
            self.selectedShape.moveBy(QPointF(1.0, 0))
        elif direction == 'up' and not self.moveOutOfBound(QPointF(0, -1.0)):
            self.selectedShape.moveBy(QPointF(0, -1.0))
        elif direction == 'down' and not self.moveOutOfBound(QPointF(0, 1.0)):
            self.selectedShape.moveBy(QPointF(0, 1.0))
        if self.selectedShape.shape_type == 'rectangle':
            w,h = self.selectedShape.size()
            if direction == 'leftEOut' and not self.moveOutOfBound(QPointF(-1.0, 0)):
                self.selectedShape.moveVertexBy(0, QPointF(-1.0, 0))
            elif direction == 'rightEOut' and not self.moveOutOfBound(QPointF(1.0, 0)):
                self.selectedShape.moveVertexBy(1, QPointF(1.0, 0))
            elif direction == 'upEOut' and not self.moveOutOfBound(QPointF(0, -1.0)):
                self.selectedShape.moveVertexBy(0, QPointF(0, -1.0))
            elif direction == 'downEOut' and not self.moveOutOfBound(QPointF(0, 1.0)):
                self.selectedShape.moveVertexBy(1, QPointF(0, 1.0))
            elif direction == 'leftEIn' and (w + 1.0 >=10):
                self.selectedShape.moveVertexBy(0, QPointF(1.0, 0))
            elif direction == 'rightEIn' and (w - 1.0 >=10):
                self.selectedShape.moveVertexBy(1, QPointF(-1.0, 0))
            elif direction == 'upEIn' and (h + 1.0 >=10):
                self.selectedShape.moveVertexBy(0, QPointF(0, 1.0))
            elif direction == 'downEIn' and (h - 1.0 >=10):
                self.selectedShape.moveVertexBy(1, QPointF(0, -1.0))
            elif direction == 'expandOut' and (not self.moveOutOfBound(QPointF(-1.0, -1.0)) or\
                                               not self.moveOutOfBound(QPointF(1.0, 1.0)) ):
                self.selectedShape.moveVertexBy(0, QPointF(-1.0, -1.0))
                self.selectedShape.moveVertexBy(1, QPointF(1.0, 1.0))
            elif direction == 'expandIn'and not((h - 1.0 <10) or (h + 1.0 <10)) and\
                                            not((w - 1.0 <10) or (w + 1.0 <10)):
                self.selectedShape.moveVertexBy(0, QPointF(1.0, 1.0))
                self.selectedShape.moveVertexBy(1, QPointF(-1.0, -1.0))
        self.shapeMoved.emit()
        self.repaint()

    def setLastLabel(self, text):
        assert text
        self.shapes[-1].label = text
        self.shapesBackups.pop()
        self.storeShapes()
        return self.shapes[-1]

    def undoLastLine(self):
        assert self.shapes
        self.current = self.shapes.pop()
        self.current.setOpen()
        if self.createMode in ['polygon', 'polyline']:
            self.line.points = [self.current[-1], self.current[0]]
        elif self.createMode in ['rectangle', 'line', 'circle']:
            self.current.points = self.current.points[0:1]
        elif self.createMode == 'point':
            self.current = None
        self.drawingPolygon.emit(True)

    def undoLastPoint(self):
        if not self.current or self.current.isClosed():
            return
        self.current.popPoint()
        if len(self.current) > 0:
            self.line[0] = self.current[-1]
        else:
            self.current = None
            self.drawingPolygon.emit(False)
        self.repaint()

    def loadPixmap(self, pixmap):
        self.pixmap = pixmap
        self.shapes = []
        self.repaint()

    def loadShapes(self, shapes):
        self.shapes = list(shapes)
        self.storeShapes()
        self.current = None
        self.repaint()

    def setShapeVisible(self, shape, value):
        self.visible[shape] = value
        self.repaint()

    def overrideCursor(self, cursor):
        self.restoreCursor()
        self._cursor = cursor
        QApplication.setOverrideCursor(cursor)

    def restoreCursor(self):
        QApplication.restoreOverrideCursor()

    def resetState(self):
        self.restoreCursor()
        self.pixmap = None
        self.shapesBackups = []
        self.update()

class MainWindow(QMainWindow):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))
    frameUpdated = Signal(int, int)

    def __init__(self, platformName, defaultFilename=None, defaultPrefdefClassFile=None, defaultSaveDir=None):
        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)
        self.setWindowIcon(QIcon(__appIcon__))
        self.colorDialog = ColorDialog(parent=self)

        self.canvas = Canvas(parent=self)
        self.canvas.zoomRequest.connect(self.zoomRequest)
        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.shapeMoved)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.frameUpdated.connect(self.updatePixmap)

        self.zoomWidget = ZoomWidget()
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

#        self.labelwidget = LabelWidget()

        self.imageditor = ImageEditor(parent=self)

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)

        self.scrollBars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal: scroll.horizontalScrollBar()
        }
        self.scrollBars[Qt.Vertical].valueChanged.connect(self.updateScrollPos)
        self.scrollBars[Qt.Horizontal].valueChanged.connect(self.updateScrollPos)

        self.zoomMode = self.MANUAL_ZOOM
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }
        self.scrollArea = scroll

        self.image = QImage()
        self.cvimage = None
        self.video = None
        self.videobuffer = []
        self.imagefiles = []
        self.dirPath = None
        self.frameNum = None
        self.totalframes = None
        self.filePath = None
        self.lineColor = None
        self.fillColor = None
        self.play_status = False
        self.skipStepValue = int()
        self.previous_Scroll_pos = None

        action = functools.partial(newAction,self)

        Quit = action('Quit', self.close,
            'Ctrl+Q', 'quit', 'Quit application', )

        Openfolder  = action('Open Folder', lambda: self.openFile(filetype='image-folder'),
            'Ctrl+Shift+O', 'open-folder', 'Open Image Folder')

        OpenVideofile  = action('Open Video', lambda: self.openFile(filetype='video-file'),
            'Ctrl+Shift+V', 'open-video-file', 'Open Video File')

        OpenImagefile = action('Open Image', lambda: self.openFile(filetype='image-file'),
            'Ctrl+Shift+I', 'open-image-file', 'Open Image File')

        Close  = action('Close File', self.closeFile,
            'Ctrl+Shift+W', 'close', 'Close File', enabled=False)

        aboutqtact = action('About &Qt', QApplication.aboutQt)

        button = functools.partial(newButton,self)

        createPolygonMode = button('Polygons', lambda: self.toggleDrawMode(False, createMode='polygon'),
            'P', 'polygon', 'Draw polygons', enabled=False,)

        createCubeMode = button('Cuboid', lambda: self.toggleDrawMode(False, createMode='cube'),
            'C', 'cuboid', 'Draw Cuboid', enabled=False,)

        createRectangleMode = button('Rectangle', lambda: self.toggleDrawMode(False, createMode='rectangle'),
            'R', 'rectangle', 'Draw rectangles', enabled=False,)

        createCircleMode = button('Circle', lambda: self.toggleDrawMode(False, createMode='circle'),
            'Ctrl+C', 'circle', 'Draw circles', enabled=False,)

        createLineMode = button('Line', lambda: self.toggleDrawMode(False, createMode='line'),
            'L', 'line', 'Draw lines', enabled=False,)

        createPointMode = button('Point', lambda: self.toggleDrawMode(False, createMode='point'),
            'Ctrl+P', 'dot', 'Draw points', enabled=False,)

        createPolyLineMode = button('PolyLine', lambda: self.toggleDrawMode(False, createMode='polyline'),
            'Ctrl+L', 'polyline', 'Draw polyline. Ctrl+LeftClick ends creation.', enabled=False,)

        editMode = button('Edit', self.setEditMode,
            'Ctrl+E', 'edit', 'Move and Edit', enabled=False,)

        shapeLineColor = button('Line Color', self.chshapeLineColor,
            icon='pen-color', tip='Change the line color for this specific shape', enabled=False)

        shapeFillColor = button('Fill Color', self.chshapeFillColor,
            icon='fill-color', tip='Change the fill color for this specific shape', enabled=False)

        delete = button('Delete', self.deleteSelectedShape,
            'Delete', 'delete', 'Delete selected Shape', enabled=False)

        undo = button('Undo', self.undoDeletetion,
            'Ctrl+Z', 'undo', 'Undo', enabled=False)

        imagedit = button('image setting', self.openImagEditor,
            icon='image-setting', tip='Edit image', enabled=False)

        fitWindow_button = button('&Fit-To-Canvas', self.setFitWindow,'Tab', 'expand',
                           'Fit to canvas size',style=Qt.ToolButtonTextBesideIcon, checkable=True, enabled=False)

        play_button = button('&Play', self.Play,'Space', 'play',
                           'Start playing video',stylesheet="border-radius: 4px;",IconSize=QSize(30,30))

        previous_button = button('&Previous', self.loadPreviousframe,'A', 'previous',
                           'Go to previous frame',stylesheet="border-radius: 4px;", IconSize=QSize(25,25))

        next_button = button('&Next', self.loadNextframe,'D', 'next',
                           'Go to next frame',stylesheet="border-radius: 4px;",IconSize=QSize(25,25))

        usePrevious_botton = button('&Use Previous', self.loadPreviousFrameShapes,'I', 'use-previous',
                           'Use previous frame shapes',stylesheet="border-radius: 4px;", IconSize=QSize(25,25))

        full_screen = button('&Go full screen', self.toggleFullscreen,'Ctrl+Tab' ,'full-screen',
                             'Go full screen',)

        menu =(
            createRectangleMode,
            createPolyLineMode,
            createPolygonMode,
            createCircleMode,
            createCubeMode,
            createLineMode,
            createPointMode,
            shapeLineColor,
            shapeFillColor,
            editMode,
            delete,
            undo,
            imagedit,
        )

        onLoadActive =(
            Close,
            createPolygonMode,
            createRectangleMode,
            createCircleMode,
            createCubeMode,
            createLineMode,
            createPointMode,
            createPolyLineMode,
            editMode,
            imagedit,
        )

        self.actions = struct(
                OpenImagefile=OpenImagefile, Quit=Quit,
                OpenVideofile=OpenVideofile, Openfolder=Openfolder, Close=Close,
                aboutqtact=aboutqtact, onLoadActive=onLoadActive
                )

        self.buttons = struct(
                createCubeMode=createCubeMode, createPolygonMode=createPolygonMode,
                createCircleMode=createCircleMode, createRectangleMode=createRectangleMode,
                createLineMode=createLineMode, createPointMode=createPointMode,
                createPolyLineMode=createPolyLineMode, menu=menu, editMode=editMode,
                shapeLineColor=shapeLineColor, shapeFillColor=shapeFillColor,
                fitWindow_button=fitWindow_button, play_button=play_button, delete=delete,
                previous_button=previous_button, next_button=next_button, undo=undo,
                usePrevious_botton=usePrevious_botton, full_screen=full_screen
                )

        self.frameinfo = QSpinBox()
        self.frameinfo.setSingleStep(1)
        self.frameinfo.setRange(0,0)
        self.frameinfo.setAlignment(Qt.AlignCenter)
        self.frameinfo.setSuffix(' / 0')
        self.frameinfo.setStatusTip('Go to a frame')
        self.frameinfo.valueChanged.connect(self.loadFrame)

        self.skipstep = QSpinBox()
        self.skipstep.setValue(1)
        self.skipstep.setRange(1,1)
        self.skipstep.setAlignment(Qt.AlignHCenter)
        self.skipstep.setSuffix(' frame')

        self.fpsbox = QComboBox()
        self.fpsbox.addItems(['max', '20', '15', '10', '5', '1'])
        self.fpsbox.setMaximumSize(QSize(45,20))
        self.fpsbox.setToolTip('Frame rate')
        self.fpsbox.setStatusTip('Frame rate')

        self.videoslider = QSlider(Qt.Horizontal)
        self.videoslider.setValue(0)
        self.videoslider.setTickPosition(QSlider.TicksBelow)
        self.videoslider.setTickInterval(1)
        self.videoslider.valueChanged.connect(self.loadFrame)
        self.videoslider.setStyleSheet('''
                             QStatusBar::separator{
                                background-color: white;
                              }
                              QSlider::groove:horizontal {
                                border: 1px solid white;
                                height: 3px;
                                background: lightgray;
                                border-radius: 2px;
                             }
                             QSlider::handle:horizontal {
                                background: rgba(250,50,50,1.0);
                                border: none;
                                width: 8px;
                                height: 6px;
                                margin: -3px 0px;
                                border-radius: 4px;
                             }
                             QSlider::handle:horizontal:pressed {
                                background-color: white;
                             }
                             QSlider::handle:horizontal:hover {
                                background-color: white;
                             }
                             QSlider::sub-page:horizontal {
                                background: rgba(250,50,50,1.0);
                                height: 3px;
                                margin: 1px 0;
                                border-radius: 1px;
                             }''')
        self.videoslider.setHidden(True)

        self.controls = QHBoxLayout()
        self.controls.setContentsMargins(0,0,5,0)
        self.controls.setAlignment(Qt.AlignLeft)

        self.controls.addWidget(self.buttons.previous_button)
        self.controls.addWidget(self.buttons.play_button)
        self.controls.addWidget(self.buttons.next_button)
        self.controls.addWidget(self.buttons.usePrevious_botton)
        self.controls.addWidget(self.fpsbox)
#        self.controls.addWidget(self.skipstep)
        self.controls.addWidget(self.videoslider)

        menubar = self.menuBar()
        menubar.setMaximumHeight(20)
        fileMenu = menubar.addMenu('&File')
        addActions(fileMenu, [self.actions.OpenImagefile, self.actions.OpenVideofile,
                              self.actions.Openfolder, self.actions.Close, self.actions.Quit])

        helpmenu = menubar.addMenu('&Help')
        addActions(helpmenu, [self.actions.aboutqtact])

        toolbar = QToolBar('Quick Access')
        toolbar.setMaximumHeight(30)
        toolbar.setContentsMargins(0,0,0,0)
        toolbar.setFloatable(True)
        toolbar.setMovable(False)
#        toolbar.setHidden(True)
#        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        addActions(toolbar, [self.actions.OpenImagefile, self.actions.OpenVideofile,
                             self.actions.Openfolder, self.actions.Quit])
        self.addToolBar(toolbar)

        self.labelCoordinates = QLabel('')
        self.progressbar = QProgressBar()
        self.progressbar.setHidden(True)
        self.progressbar.setMaximumSize(QSize(150, 16))
        self.timer = QBasicTimer()

        self.statusBar().addPermanentWidget(self.progressbar)
        self.statusBar().addPermanentWidget(self.labelCoordinates)
        self.statusBar().addPermanentWidget(QLabel('<b>Skip:</b>'))
        self.statusBar().addPermanentWidget(self.skipstep)
        self.statusBar().addPermanentWidget(QLabel('<b>Frame:</b>'))
        self.statusBar().addPermanentWidget(self.frameinfo)
        self.statusBar().addPermanentWidget(QLabel('<b>Zoom:</b>'))
        self.statusBar().addPermanentWidget(self.zoomWidget)
        self.statusBar().addPermanentWidget(self.buttons.fitWindow_button)
        self.statusBar().addPermanentWidget(self.buttons.full_screen)
        self.statusBar().setMaximumHeight(25)
#        self.setStyleSheet('''
#                              QProgressBar {
#                                border-radius: 5px;
#                                border: 2px solid grey;
#                                text-align: center;
#                                margin: 1px 0;
#                              }
#                              QProgressBar::chunk {
#                                background: qlineargradient( x1:0 y1:0, x2:1 y2:0, stop:0 hotpink, stop:1 cornflowerblue);
#                              }
#                              QStatusBar {
#                                background-color: white;
#                                color: black;
#                              }
#                              ''')

        options = QVBoxLayout()
        addWidgets(options, menu)
#        for option in menu:
#              option.setStyleSheet("border-radius: 4px;")
        options.setAlignment(Qt.AlignHCenter)
        options.setContentsMargins(0,0,0,0)
        print(options.spacing())

        jointlayout = QHBoxLayout()
        jointlayout.setAlignment(Qt.AlignHCenter)
        jointlayout.setContentsMargins(0,0,0,0)
        jointlayout.addLayout(options)
        jointlayout.addWidget(self.scrollArea)

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(5,0,0,0)
#        self.layout.addWidget(self.scrollArea)
        self.layout.addLayout(jointlayout)
        self.layout.addLayout(self.controls)

        window = QWidget()
        window.setLayout(self.layout)
        self.setCentralWidget(window)

#        self.propertydock = QDockWidget("Property" ,self)
#        self.dockFeatures = QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetFloatable
#        self.propertydock.setFeatures(self.propertydock.features() ^ self.dockFeatures)

#        self.addDockWidget(Qt.RightDockWidgetArea ,self.propertydock)
#        self.propertydock.setWidget(self.labelwidget)

        self.zoomWidget.setEnabled(False)
        self.frameinfo.setEnabled(False)
        self.skipstep.setEnabled(False)
        self.setControlsHidden(True)
        self.showMaximized()
#        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint)


    def closeFile(self):
        pass

    def openFile(self, filetype='video-file'):
        if filetype == 'video-file':
            filename = QFileDialog.getOpenFileName(self, '%s - Open video file' % __appname__, '',
                'Video Files (*.avi *.mp4)',None, QFileDialog.DontUseNativeDialog)[0]
            if os.path.isfile(filename):
#                self.queueEvent(functools.partial(self.loadFile, os.path.abspath(filename) or ""))
                self.loadFile(os.path.abspath(filename))

        elif filetype == 'image-file':
            filename = QFileDialog.getOpenFileName(self, '%s - Open image file' % __appname__, '',
                'Image Files (*.jpg *.jpeg *.jpe *.jp2 *.png *.bmp *.tif *.tiff)', None, QFileDialog.DontUseNativeDialog)[0]
            if os.path.isfile(filename):
#                self.queueEvent(functools.partial(self.loadFile, os.path.abspath(filename) or ""))
                self.loadFile(os.path.abspath(filename))

        elif filetype == 'image-folder':
            targetDirPath = str(QFileDialog.getExistingDirectory(
                    self, '%s - Open Folder' % __appname__,'',
                    QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks | QFileDialog.DontUseNativeDialog))
            if os.path.isdir(targetDirPath):
                self.loadDirImages(targetDirPath)

    def closeEvent(self, event):
        self.reply = QMessageBox.question(self, 'Confirm Exit',
            "Are you sure to Quit ?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if self.reply == QMessageBox.Yes:
            if self.imageditor.isVisible():
                self.imageditor.close()
            if self.video and  self.video.isOpened():
                self.video.release()
                cv2.destroyAllWindows()

            event.accept()
        else:
            event.ignore()

    def queueEvent(self, function):
        QTimer.singleShot(0, function)

    def openImagEditor(self):
        self.imageditor.show()
        pass

    def convertToQImage(self, imageData):
        if type(imageData) == np.ndarray:
            image = np.copy(imageData)
            if len(image.shape) < 3 or image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            height, width, byteValue = image.shape
            byteValue = byteValue * width

            return QImage(image, width, height, byteValue, QImage.Format_RGB888)

        elif type(imageData) == bytes:
            return QImage.fromData(imageData)

        return QImage()

    def readVideo(self, FilePath):
        return cv2.VideoCapture(FilePath)

    def loadPreviousFrameShapes(self):
        print("use previous doesnot work")

    def updatePixmap(self, current, total):
        if not current == self.frameinfo.value():
            self.frameinfo.setValue(current)
        if not current == self.videoslider.value():
            self.videoslider.setValue(current)

        if self.videobuffer:
            self.cvimage = self.videobuffer[current]
        elif self.imagefiles:
            self.cvimage = cv2.imread(self.imagefiles[current])
        self.image = self.convertToQImage(self.cvimage)

        pass

    def isCompatible(self, file, filetype='img'):
        if filetype == 'img':
            compatible_file_formats = [".jpg", ".jpeg", ".jpe", ".jp2", ".png", ".bmp", ".tif", ".tiff"]
        elif filetype == 'vid':
            compatible_file_formats = [".mp4", ".avi"]
        else:
            assert False, "unsupported filetype"

        name,ext = os.path.splitext(os.path.split(file)[1])
        compatible_file_formats = compatible_file_formats + [file_format.upper() for file_format in compatible_file_formats]
        if ext in compatible_file_formats:
            return True
        return False

    def timerEvent(self, event):
        if not self.progressbar.isHidden() and self.value >= 100:
            self.timer.stop()
            self.progressbar.setHidden(True)
            self.progressbar.setValue(0)
        if not self.progressbar.isHidden():
            self.progressbar.setValue(int(self.value))

    def loadFile(self, Path):
        self.filePath = Path
        self.image = QImage()
        self.resetState()
        self.loadPixmapToCanvas()
        self.canvas.setEnabled(False)
        if self.isCompatible(Path, filetype = 'vid') and self.video is None:
            self.video = self.readVideo(Path)
            self.videobuffer = []
            self.temp = []
            self.imagefiles = []
            self.totalframes = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.totalframes > 2000:
                QMessageBox.critical(self, 'Error',"Larger videos  are not supported in this version")
                self.video.release()
                cv2.destroyAllWindows()
                self.loadPixmapToCanvas()
                self.paintCanvas()
                return
            self.frameNum = 0
            self.progressbar.setHidden(False)
            self.timer.start(100, self)
            while self.video.isOpened():
                self.video.set(15, self.frameNum)
                ret, frame = self.video.read()
                self.value = (self.frameNum/self.totalframes)*100
                QCoreApplication.processEvents()
                if ret is True:
                    if len(self.temp) >= 500:
                        self.videobuffer += self.temp
                        self.temp = []
                    self.temp.append(frame)
                    self.frameNum += 1
                else:
                    self.videobuffer += self.temp
                    del self.temp
                    break
            self.video.release()
            cv2.destroyAllWindows()
            self.video = None
            self.setControlsHidden(False)
            self.frameNum, self.totalframes = 0, len(self.videobuffer)
            self.frameUpdated.emit(self.frameNum, self.totalframes-1)
            self.videoslider.setRange(self.frameNum, self.totalframes-1)
            self.skipstep.setRange(self.frameNum+1, self.totalframes-1)
            self.frameinfo.setRange(self.frameNum, self.totalframes-1)
            self.frameinfo.setSuffix(' / {}'.format(self.totalframes-1))
#            self.videoslider.setHidden(False)

        elif self.isCompatible(Path,filetype = 'img'):
            self.setControlsHidden(True)
            self.frameNum, self.totalframes = 0, 1
            self.frameUpdated.emit(0, 1)
            self.cvimage = cv2.imread(Path)
            self.image = self.convertToQImage(self.cvimage)
#            self.videoslider.setHidden(True)

        else:
            assert False,"unsupported image"
        self.loadPixmapToCanvas()
        self.paintCanvas()

    def loadDirImages(self, dirpath, pattern=None, load=True):
        files = os.listdir(dirpath)
        self.dirPath = dirpath
        filteredfiles = []
        #formats = [".%s"%fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        for  file in files:
            if self.isCompatible(file,filetype = 'img'):
                filteredfiles.append(file)

        files_sortkeys = natsort.natsort_keygen(alg = natsort.ns.INT)
        filteredfiles.sort(key = files_sortkeys)
        self.imagefiles = []
        self.videobuffer = []
        for file in filteredfiles:
            imagepath = os.path.abspath(os.path.join(dirpath,file))
            if os.path.isfile(imagepath):
                self.imagefiles.append(imagepath)

        self.frameNum, self.totalframes = 0, len(self.imagefiles)

        if not self.totalframes == 0:
            self.setControlsHidden(False)
            self.frameUpdated.emit(self.frameNum, self.totalframes-1)
            self.videoslider.setRange(self.frameNum,self.totalframes-1)
            self.frameinfo.setRange(self.frameNum, self.totalframes-1)
            self.skipstep.setRange(self.frameNum+1,self.totalframes-1)
            self.frameinfo.setSuffix(' / {}'.format(self.totalframes-1))
#            self.videoslider.setHidden(False)
        else:
            self.setControlsHidden(True)
            self.frameUpdated.emit(0, 1)
#            self.videoslider.setHidden(True)
            QMessageBox.warning(self, 'Warning', "No images found")
        self.loadPixmapToCanvas()
        self.paintCanvas()

    def loadPixmapToCanvas(self):
        self.canvas.loadPixmap(QPixmap.fromImage(self.image))
        if not self.image.isNull():
            self.toggleActions(True)
            value = self.scalers[self.FIT_WINDOW]()
            self.zoomWidget.setRange(100*value,self.zoomWidget.maximum())
        else:
            self.toggleActions(False)

    def setControlsHidden(self, hide=True):
        if hide:
            self.layout.removeItem(self.controls)
        else:
            self.layout.addLayout(self.controls)

        self.buttons.previous_button.setVisible(not hide)
        self.buttons.play_button.setVisible(not hide)
        self.buttons.next_button.setVisible(not hide)
        self.buttons.usePrevious_botton.setVisible(not hide)
        self.fpsbox.setVisible(not hide)
        self.videoslider.setVisible(not hide)

    def loadFrame(self, value):
        if self.play_status is True:
            return
        self.frameNum = value
        self.frameUpdated.emit(value, self.totalframes-1)
        self.loadPixmapToCanvas()

    def loadNextframe(self):
        canload = (self.videobuffer or self.imagefiles) and (self.frameNum+1 < self.totalframes)
        if canload:
            step = self.skipstep.value()
            self.frameNum += step
            self.frameUpdated.emit(self.frameNum, self.totalframes-1)
            self.loadPixmapToCanvas()
            if self.imageditor.isVisible():
                self.imageditor.updateSettings()
            return True
        return False

    def loadPreviousframe(self):
        canload = (self.videobuffer or self.imagefiles) and (self.frameNum-1 >= 0)
        if canload:
            step = self.skipstep.value()
            self.frameNum -= step
            self.frameUpdated.emit(self.frameNum, self.totalframes-1)
            self.loadPixmapToCanvas()
            if self.imageditor.isVisible():
                self.imageditor.updateSettings()
            return True
        return False


    def togglePlayMode(self, status = True):
        self.videoslider.setEnabled(not status)
        self.buttons.previous_button.setEnabled(not status)
        self.buttons.next_button.setEnabled(not status)
        self.buttons.usePrevious_botton.setEnabled(not status)
        self.fpsbox.setEnabled(not status)
        self.frameinfo.setEnabled(not status)
        if status is True:
            self.buttons.play_button.setIcon(newIcon('pause'))
        else:
            self.buttons.play_button.setIcon(newIcon('play'))

    def toggleDrawingSensitive(self, drawing=True):
        """In the middle of drawing, toggling between modes should be disabled."""
        self.canvas.setEditing(not drawing)
        self.buttons.editMode.setEnabled(not drawing)
        self.buttons.undo.setEnabled(not drawing)
        self.buttons.delete.setEnabled(not drawing)
        self.buttons.createPolygonMode.setEnabled(not drawing)
        self.buttons.createRectangleMode.setEnabled(not drawing)
        self.buttons.createCircleMode.setEnabled(not drawing)
        self.buttons.createLineMode.setEnabled(not drawing)
        self.buttons.createCubeMode.setEnabled(not drawing)
        self.buttons.createPointMode.setEnabled(not drawing)
        self.buttons.createPolyLineMode.setEnabled(not drawing)
        if not drawing:
            self.toggleDrawMode(edit= drawing,createMode=self.canvas.createMode)

    def toggleDrawMode(self, edit=True, createMode='polygon'):
        self.canvas.setEditing(edit)
        self.canvas.createMode = createMode
        if edit:
            self.buttons.createPolygonMode.setEnabled(True)
            self.buttons.createRectangleMode.setEnabled(True)
            self.buttons.createCircleMode.setEnabled(True)
            self.buttons.createLineMode.setEnabled(True)
            self.buttons.createPointMode.setEnabled(True)
            self.buttons.createCubeMode.setEnabled(True)
            self.buttons.createPolyLineMode.setEnabled(True)
        else:
            if createMode == 'polygon':
                self.buttons.createPolygonMode.setEnabled(False)
                self.buttons.createRectangleMode.setEnabled(True)
                self.buttons.createCircleMode.setEnabled(True)
                self.buttons.createLineMode.setEnabled(True)
                self.buttons.createPointMode.setEnabled(True)
                self.buttons.createCubeMode.setEnabled(True)
                self.buttons.createPolyLineMode.setEnabled(True)
            elif createMode == 'rectangle':
                self.buttons.createPolygonMode.setEnabled(True)
                self.buttons.createRectangleMode.setEnabled(False)
                self.buttons.createCircleMode.setEnabled(True)
                self.buttons.createLineMode.setEnabled(True)
                self.buttons.createPointMode.setEnabled(True)
                self.buttons.createCubeMode.setEnabled(True)
                self.buttons.createPolyLineMode.setEnabled(True)
            elif createMode == 'line':
                self.buttons.createPolygonMode.setEnabled(True)
                self.buttons.createRectangleMode.setEnabled(True)
                self.buttons.createCircleMode.setEnabled(True)
                self.buttons.createLineMode.setEnabled(False)
                self.buttons.createPointMode.setEnabled(True)
                self.buttons.createCubeMode.setEnabled(True)
                self.buttons.createPolyLineMode.setEnabled(True)
            elif createMode == 'point':
                self.buttons.createPolygonMode.setEnabled(True)
                self.buttons.createRectangleMode.setEnabled(True)
                self.buttons.createCircleMode.setEnabled(True)
                self.buttons.createLineMode.setEnabled(True)
                self.buttons.createPointMode.setEnabled(False)
                self.buttons.createCubeMode.setEnabled(True)
                self.buttons.createPolyLineMode.setEnabled(True)
            elif createMode == 'circle':
                self.buttons.createPolygonMode.setEnabled(True)
                self.buttons.createRectangleMode.setEnabled(True)
                self.buttons.createCircleMode.setEnabled(False)
                self.buttons.createLineMode.setEnabled(True)
                self.buttons.createPointMode.setEnabled(True)
                self.buttons.createCubeMode.setEnabled(True)
                self.buttons.createPolyLineMode.setEnabled(True)
            elif createMode == 'polyline':
                self.buttons.createPolygonMode.setEnabled(True)
                self.buttons.createRectangleMode.setEnabled(True)
                self.buttons.createCircleMode.setEnabled(True)
                self.buttons.createLineMode.setEnabled(True)
                self.buttons.createPointMode.setEnabled(True)
                self.buttons.createCubeMode.setEnabled(True)
                self.buttons.createPolyLineMode.setEnabled(False)
            elif createMode == 'cube':
                self.buttons.createPolygonMode.setEnabled(True)
                self.buttons.createRectangleMode.setEnabled(True)
                self.buttons.createCircleMode.setEnabled(True)
                self.buttons.createLineMode.setEnabled(True)
                self.buttons.createPointMode.setEnabled(True)
                self.buttons.createCubeMode.setEnabled(False)
                self.buttons.createPolyLineMode.setEnabled(True)
            else:
                raise ValueError('Unsupported createMode: %s' % createMode)
        self.buttons.editMode.setEnabled(not edit)

    def chshapeLineColor(self):
        color = self.colorDialog.getColor(
            self.lineColor, 'Choose line color', default=DEFAULT_LINE_COLOR)
        if color:
            self.canvas.selectedShape.line_color = color
            self.canvas.update()
#            self.setDirty()

    def chshapeFillColor(self):
        color = self.colorDialog.getColor(
            self.fillColor, 'Choose fill color', default=DEFAULT_FILL_COLOR)
        if color:
            self.canvas.selectedShape.fill_color = color
            self.canvas.update()
#            self.setDirty()

    def setEditMode(self):
        self.toggleDrawMode(True)

    def Play(self):
        canload = (self.videobuffer or self.imagefiles) and not self.play_status
        selection = self.fpsbox.currentText()
        fps = int(selection) if not selection == 'max' else 100
        fconst = 1/int(round((5*(fps/2))))
        if canload:
            self.play_status = True
            self.togglePlayMode(True)
            if self.videobuffer:
                source = self.videobuffer[self.frameNum:]
            elif self.imagefiles:
                source = self.imagefiles[self.frameNum:]
            transfertobuffer = self.frameNum == 0
            for frame in source:
                fstart = time.time()
                self.cvimage = frame
                if type(frame) is str:
                    self.cvimage = cv2.imread(frame)
                    if transfertobuffer is True:
                        self.videobuffer.append(self.cvimage)
                QCoreApplication.processEvents()
                self.image = self.convertToQImage(self.cvimage)
                self.loadPixmapToCanvas()

                if self.imageditor.isVisible():
                    self.imageditor.updateSettings()
                if self.play_status is False:
                    self.togglePlayMode(False)
                    if self.imagefiles:
                        self.videobuffer = []
                    return
                self.frameNum += 1
                if not abs(fstart - time.time()) >= fconst:
                    time.sleep(abs(fstart - time.time() - fconst))
                self.frameinfo.setValue(self.frameNum)
                self.videoslider.setValue(self.frameNum)
                QCoreApplication.processEvents()
            self.frameNum = 0
            self.frameinfo.setValue(self.frameNum)
            self.videoslider.setValue(self.frameNum)
            self.play_status = False
            self.togglePlayMode(False)
            if self.imagefiles and len(self.imagefiles) == len(self.videobuffer):
                self.imagefiles = []
        else:
            self.play_status = False
            self.togglePlayMode(False)

    def toggleFullscreen(self):
        if self.isFullScreen() is True:
            self.showMaximized()
        else:
            self.showFullScreen()

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        self.zoomWidget.setEnabled(value)
        if self.videobuffer or self.imagefiles:
            self.skipstep.setEnabled(value)
            self.frameinfo.setEnabled(value)
        else:
            self.skipstep.setEnabled(False)
            self.frameinfo.setEnabled(False)
        self.buttons.fitWindow_button.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    # React to canvas signals.
    @Slot(bool)
    def shapeSelectionChanged(self, selected=False):
        self.buttons.delete.setEnabled(selected)
        self.buttons.shapeLineColor.setEnabled(selected)
        self.buttons.shapeFillColor.setEnabled(selected)

    @Slot()
    def newShape(self):
        pass
#            print("hhh")

    def shapeMoved(self):
        for shape in self.canvas.shapes:
            points = shape.points
            pts = []
            for point in points:
                pts.append([point.x(), point.y()])
            print(pts)

    def deleteSelectedShape(self):
        deletedShape = self.canvas.deleteSelected()
        if deletedShape is not None:
#            print(deletedShape.Id,"deleted")
            pass

    def undoDeletetion(self):
#        self.canvas.undoDeletion()
#        self.canvas.deSelectShape()
        pass

    def resetState(self):
        self.filePath = None
        self.imageData = None
        self.labelFile = None
        self.canvas.resetState()
        self.labelCoordinates.clear()

    def resizeEvent(self, event):
        try:
            if self.canvas and not self.image.isNull()\
               and self.zoomMode != self.MANUAL_ZOOM:
                self.adjustScale()
        except AttributeError:
            pass
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        if not self.image.isNull():
            self.canvas.scale = 0.01 * self.zoomWidget.value()
            self.canvas.epsilon = 10.0 /self.canvas.scale if self.canvas.scale >= 1.0 else 10.0
            self.canvas.adjustSize()
            self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        self.zoomWidget.setRange(100*self.scalers[self.FIT_WINDOW](),self.zoomWidget.maximum())
        self.zoomWidget.setValue(100 * value)

    def scaleFitWindow(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 4.0  # So that no scrollbars are generated.
        w1 = self.scrollArea.width() - e
        h1 = self.scrollArea.height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width()
        h2 = self.canvas.pixmap.height()
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.scrollArea.width() - 2.0
        return w / self.canvas.pixmap.width()

    def scrollRequest(self, delta, orientation):
        units = - delta / (8 * 15)
        bar = self.scrollBars[orientation]
        bar.setValue(bar.value() + bar.singleStep() * units)

    def updateScrollPos(self):
        self.previous_Scroll_pos = self.scrollBars[Qt.Horizontal].value(),self.scrollBars[Qt.Vertical].value()

    def setScrollbarPos(self):
        self.scrollBars[Qt.Horizontal].setValue(self.previous_Scroll_pos[0])
        self.scrollBars[Qt.Vertical].setValue(self.previous_Scroll_pos[1])

    def setZoom(self, value):
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)

    def addZoom(self, increment=10):
        self.setZoom(self.zoomWidget.value() + increment)

    @Slot(int, QPoint)
    def zoomRequest(self, delta,pos):
        canvas_width_old = self.canvas.width()
        units = 1
        if delta < 0 :
            units = -1
        step = 10
        if not self.zoomWidget.value() == self.zoomWidget.minimum():
            step = (self.zoomWidget.value() - self.zoomWidget.minimum())/10

        self.addZoom(units*step)

#        self.zoomWidget.setSingleStep(step)

        canvas_width_new = self.canvas.width()
        if canvas_width_old != canvas_width_new:
            canvas_scale_factor = canvas_width_new / canvas_width_old

            x_shift = round(pos.x() * canvas_scale_factor) - pos.x()
            y_shift = round(pos.y() * canvas_scale_factor) - pos.y()
            self.scrollBars[Qt.Horizontal].setValue(self.scrollBars[Qt.Horizontal].value() + x_shift)
            self.scrollBars[Qt.Vertical].setValue(self.scrollBars[Qt.Vertical].value() + y_shift)

    def setFitWindow(self, value=True):
        self.buttons.fitWindow_button.setChecked(value)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

class ZoomWidget(QSpinBox):
    def __init__(self, value=100):
        super(ZoomWidget, self).__init__()
#        self.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.setRange(1, 10000)
        self.setSingleStep(20)
        self.setSuffix(' %')
        self.setValue(value)
        self.setToolTip(u'Zoom Level')
        self.setStatusTip(self.toolTip())
        self.setAlignment(Qt.AlignCenter)

    def minimumSizeHint(self):
        height = super(ZoomWidget, self).minimumSizeHint().height()
        fm = QFontMetrics(self.font())
        width = fm.width(str(self.maximum()))
        return QSize(width, height)

class LabelWidget(QWidget):

    labelUpdated = Signal(dict)

    def __init__(self,*args, **kwargs):
        self.property_type = kwargs.pop('property_type', 'bbox')
        super(LabelWidget, self).__init__(*args, **kwargs)
        self.objectclass = dict()
        self.loadLabels()
        self.temp = []
        self.Id = None
        self.initWidget()

    def initWidget(self):
        self.maingrid = QGridLayout()
        self.populateGrid(self.maingrid,self.objectclass)
        self.Layout = QVBoxLayout()
        self.Layout.addLayout(self.maingrid)
        self.setLayout(self.Layout)
#        self.setFixedSize(self.sizeHint())

    def populateGrid(self, gridlayout, sourcedict):
        for i, items in  enumerate(sourcedict.items()):
            mainclass, values = items
            subclass = [list(value)[0] for value in values]
            gridlayout.addWidget(QLabel(mainclass), i, 0)
            cbox = QComboBox()
            cbox.currentTextChanged.connect(lambda:self.populatelayerOne(gridlayout, values))
            cbox.addItems(subclass)
            gridlayout.addWidget(cbox, i,1)

    def populatelayerOne(self, gridlayout, sourcedict):
        sender = self.sender()
        selectionIndex = sender.currentIndex()
        selectionText = sender.currentText()
        count, row, col = gridlayout.count(), gridlayout.rowCount(), gridlayout.columnCount()
        for i in range(1,row):
            for j in range(1,col):
                item = gridlayout.itemAtPosition(i,j)
                item.widget()
        for i,items in enumerate(sourcedict[selectionIndex][selectionText].items()):
            subsel, values = items
            row = gridlayout.rowCount()
            print(subsel, list(values))
            name = QLabel(subsel)
            gridlayout.addWidget(name, row, 1)
            cbox = QComboBox()
            cbox.currentTextChanged.connect(lambda:self.populatelayerTwo(gridlayout, values))
            cbox.addItems(values)
            gridlayout.addWidget(cbox, row,2)
            self.temp.append(name)
            self.temp.append(cbox)

    def populatelayerTwo(self, gridlayout, sourcedict):
        sender = self.sender()
        selectionIndex = sender.currentIndex()
        selectionText = sender.currentText()

    def loadLabels(self):
        if self.property_type == 'bbox':
            with open("labels/object_labels.json") as json_file:
                self.objectclass = json.load(json_file)

class ImageEditor(QDialog):
    def __init__(self,parent=None):
        super(ImageEditor, self).__init__(parent)
        self.setFixedSize(QSize(300,180))
        self.setWindowTitle("Image Enhancer")
        settingGroup = QGroupBox("Enhance Image Settings")

        brightnessLabel, brightness = QLabel("Brightness :"), QSlider(Qt.Horizontal)
        contrastLabel, contrast = QLabel("Contrast :"), QSlider(Qt.Horizontal)
        sharpnessLabel, sharpness = QLabel("Sharpness :"), QSlider(Qt.Horizontal)
        colorLabel, color = QLabel("Color :"), QSlider(Qt.Horizontal)
        edgelabel, edge = QLabel("Edge :"), QSlider(Qt.Horizontal)

        for slider,slot in zip([brightness,contrast,sharpness, color, edge], [self.updateSettings]*5):
            self.setSliderProperty(slider, slot)

        restorebutton = newButton(self, '&Restore Defaults', self.restore, None,
                                  None, 'restore', style=Qt.ToolButtonTextBesideIcon)

        self.settinglayout = QGridLayout()
        addWidgets(self.settinglayout, [(brightnessLabel,0,0), (brightness,0,1),
                                   (contrastLabel,1,0), (contrast,1,1),
                                   (sharpnessLabel,2,0), (sharpness,2,1),
                                   (colorLabel,3,0), (color,3,1), (edgelabel,4,0), (edge,4,1)])
        settingGroup.setLayout(self.settinglayout)

        layout = QVBoxLayout()
        addWidgets(layout, [settingGroup,restorebutton])
        self.settings = struct(brightness=brightness, contrast=contrast, sharpness=sharpness, color=color, edge=edge)
        self.setLayout(layout)

    def setSliderProperty(self,slider, slot):
        slider.setRange(0,200)
        slider.setValue(100)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(100)
        slider.valueChanged.connect(slot)

    @Slot()
    def updateSettings(self):
        image = window.cvimage

        if image is None:
            return

        brightnesvalue = self.settings.brightness.value()
        contrastvalue = self.settings.contrast.value()
        sharpnessvalue = self.settings.sharpness.value()
        colorvalue = self.settings.color.value()
        edgevalue = (255*self.settings.edge.value()/200)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if not int(edgevalue) == 127:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.medianBlur(image,5)
            image = cv2.Canny(gray,127,edgevalue)
            image  = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


        pilimage = Image.fromarray(image)
        pilimage = ImageEnhance.Brightness(pilimage).enhance(brightnesvalue/100)
        pilimage = ImageEnhance.Contrast(pilimage).enhance(contrastvalue/100)
        pilimage = ImageEnhance.Sharpness(pilimage).enhance(sharpnessvalue/100)
        pilimage = ImageEnhance.Color(pilimage).enhance(colorvalue/100)

        shapes = window.canvas.shapes
        window.image = ImageQt.ImageQt(pilimage)
        window.loadPixmapToCanvas()
        window.canvas.loadShapes(shapes)

    def restore(self):
        self.settings.brightness.setValue(100)
        self.settings.contrast.setValue(100)
        self.settings.sharpness.setValue(100)
        self.settings.color.setValue(100)
        self.settings.edge.setValue(100)

class ColorDialog(QColorDialog):
    def __init__(self, parent=None):
        super(ColorDialog, self).__init__(parent)
        self.setOption(QColorDialog.ShowAlphaChannel)
        # The Mac native dialog does not support our restore button.
        self.setOption(QColorDialog.DontUseNativeDialog)
        # Add a restore defaults button.
        # The default is set at invocation time, so that it
        # works across dialogs for different elements.
        self.default = None
        self.bb = self.layout().itemAt(1).widget()
        self.bb.addButton(QDialogButtonBox.RestoreDefaults)
        self.bb.clicked.connect(self.checkRestore)

    def getColor(self, value=None, title=None, default=None):
        self.default = default
        if title:
            self.setWindowTitle(title)
        if value:
            self.setCurrentColor(value)
        return self.currentColor() if self.exec_() else None

    def checkRestore(self, button):
        if self.bb.buttonRole(button) & \
                QDialogButtonBox.ResetRole and self.default:
            self.setCurrentColor(self.default)

class struct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getitem__(self, key):
        return self.__dict__[key]

def newButton(parent, text, slot=None, shortcut=None, icon=None,
              tip=None, checkable=False, enabled=True,stylesheet=None,
              style=None,IconSize=None):
    """Create a new button and assign callbacks, shortcuts, etc."""
    b = QToolButton(parent)
    if type(text) is str:
        b.setText(text)
    if icon is not None:
        b.setIcon(newIcon(icon))
    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            b.setShortcuts(shortcut)
            if tip is not None:
                b.setToolTip(tip+'\n'+"Shortcut: "+ shortcut)
                b.setStatusTip(tip)
        else:
            b.setShortcut(shortcut)
            if tip is not None:
                b.setToolTip(tip+'\n'+"Shortcut: "+ shortcut)
                b.setStatusTip(tip)
    if tip is not None and shortcut is None:
        b.setToolTip(tip)
        b.setStatusTip(tip)
    if slot is not None:
        b.clicked.connect(slot)
    if checkable:
        b.setCheckable(True)
    if stylesheet is not None:
        b.setStyleSheet(stylesheet)
    if style is not None:
        b.setToolButtonStyle(style)
    if IconSize is not None:
        b.setIconSize(IconSize)
    b.setEnabled(enabled)
    return b

def newAction(parent, text, slot=None, shortcut=None, icon=None,
              tip=None, checkable=False, enabled=True):
    """Create a new action and assign callbacks, shortcuts, etc."""
    a = QAction(text, parent)
    if icon is not None:
        a.setIconText(text.replace(' ', '\n'))
        a.setIcon(newIcon(icon))
    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            a.setShortcuts(shortcut)
            if tip is not None:
                a.setToolTip(tip+'\n'+"Shortcut: "+ shortcut)
                a.setStatusTip(tip)
        else:
            a.setShortcut(shortcut)
            if tip is not None:
                a.setToolTip(tip+'\n'+"Shortcut: "+ shortcut)
                a.setStatusTip(tip)
    if tip is not None and shortcut is None:
        a.setToolTip(tip)
        a.setStatusTip(tip)
    if slot is not None:
        a.triggered.connect(slot)
    if checkable:
        a.setCheckable(True)
    a.setEnabled(enabled)
    return a

def addActions(widget, actions):
    for action in actions:
        if action is None:
            widget.addSeparator()
        elif isinstance(action, QMenu):
            widget.addMenu(action)
        else:
            widget.addAction(action)

def addWidgets(layout, widgets):
    for widget in widgets:
        if isinstance(layout, QGridLayout):
            layout.addWidget(widget[0],*widget[1:])
        elif isinstance(widget, QWidget):
            layout.addWidget(widget)

def newIcon(icon):
    return QIcon(os.path.join(__icondir,'%s.png' % icon))

def distance(p):
    return math.sqrt(p.x() * p.x() + p.y() * p.y())

def distancetopoint(p1,p2):
    return math.sqrt((p1.x() - p2.x())**2 + (p1.y() - p2.y())**2)

def distancetoline(point, line):
    p1, p2 = line
    p1 = np.array([p1.x(), p1.y()])
    p2 = np.array([p2.x(), p2.y()])
    p3 = np.array([point.x(), point.y()])
    if np.dot((p3 - p1), (p2 - p1)) < 0:
        return np.linalg.norm(p3 - p1)
    if np.dot((p3 - p2), (p1 - p2)) < 0:
        return np.linalg.norm(p3 - p2)
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

def generateColorByText(text):
    s = text
    hashCode = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16)
    r = int((hashCode / 255) % 255)
    g = int((hashCode / 65025)  % 255)
    b = int((hashCode / 16581375)  % 255)
    return QColor(r, g, b, 100)

def read(filename, default=None):
    try:
        with open(filename, 'rb') as f:
            return f.read()
    except:
        return default

def readCSS(fname):

	f = open(fname, 'r')
	s = f.read()
	f.close()
	return s

#print(Canvas.__mro__)
if __name__ == '__main__':
    platformName = platform.system()
    app = QApplication(sys.argv)
    window = MainWindow(platformName = platformName)
#    window.setStyleSheet(readCSS("MetroDark.qss"))
    window.show()
    sys.exit(app.exec_())