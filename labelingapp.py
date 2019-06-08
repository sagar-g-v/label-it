# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 10:16:54 2019

@author: sagarg9
"""
import os
import sys
import cv2
import pandas as pd
import copy
import math
import hashlib
import functools
import numpy as np
from qtpy.QtWidgets import (QMainWindow, QFileDialog, QApplication, QWidget,
                             QLabel, QScrollBar, QMenu, QToolButton,
                             QSpinBox, QScrollArea, QSlider, QAction,
                             QPushButton, QGridLayout, QLineEdit, QComboBox,
                             QCheckBox, QCompleter, QDockWidget, QHBoxLayout,
                             QGroupBox, QVBoxLayout,QMessageBox)
from qtpy.QtCore import (QBasicTimer, Qt, QCoreApplication, QRectF, 
                         QSettings, QSize, QPointF,QPoint,Signal,QTimer,
                         Slot)
from qtpy.QtGui import (QIcon, QPicture,QPixmap, QColor, QPen,QBrush, QFont, QPainterPath,
                        QFontMetrics, QImage, QCursor, QPainter,QIntValidator)
from qtpy import QT_VERSION
QT5 = QT_VERSION[0] == '5'

# TODO(unknown):
# - [opt] Store paths instead of creating new ones at each paint.

DEFAULT_LINE_COLOR = QColor(0, 255, 0, 128)
DEFAULT_FILL_COLOR = QColor(255, 0, 0, 128)
DEFAULT_SELECT_LINE_COLOR = QColor(255, 255, 255)
DEFAULT_SELECT_FILL_COLOR = QColor(0, 128, 255, 155)
DEFAULT_VERTEX_FILL_COLOR = QColor(0, 255, 0, 255)
DEFAULT_HVERTEX_FILL_COLOR = QColor(255, 0, 0)


class Shape(object):

    P_SQUARE, P_ROUND = 0, 1

    MOVE_VERTEX, NEAR_VERTEX = 0, 1

    # The following class variables influence the drawing of all shape objects.
    line_color = DEFAULT_LINE_COLOR
    fill_color = DEFAULT_FILL_COLOR
    select_line_color = DEFAULT_SELECT_LINE_COLOR
    select_fill_color = DEFAULT_SELECT_FILL_COLOR
    vertex_fill_color = DEFAULT_VERTEX_FILL_COLOR
    hvertex_fill_color = DEFAULT_HVERTEX_FILL_COLOR
    point_type = P_ROUND
    point_size = 8
    scale = 1.0

    def __init__(self, label=None, line_color=None, shape_type=None):
        self.label = label
        self.points = []
        self.fill = False
        self.selected = False
        self.shape_type = shape_type

        self._highlightIndex = None
        self._highlightMode = self.NEAR_VERTEX
        self._highlightSettings = {
            self.NEAR_VERTEX: (4, self.P_ROUND),
            self.MOVE_VERTEX: (1.5, self.P_SQUARE),
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
           'line', 'circle', 'linestrip']:
            raise ValueError('Unexpected shape_type: {}'.format(value))
        self._shape_type = value

    def close(self):
        self._closed = True

    def addPoint(self, point):
        if self.points and point == self.points[0]:
            self.close()
        else:
            self.points.append(point)

    def popPoint(self):
        if self.points:
            return self.points.pop()
        return None

    def insertPoint(self, i, point):
        self.points.insert(i, point)

    def isClosed(self):
        return self._closed

    def setOpen(self):
        self._closed = False

    def getRectFromLine(self, pt1, pt2):
        x1, y1 = pt1.x(), pt1.y()
        x2, y2 = pt2.x(), pt2.y()
        return QRectF(x1, y1, x2 - x1, y2 - y1)

    def paint(self, painter):
        if self.points:
            color = self.select_line_color \
                if self.selected else self.line_color
            pen = QPen(color)
            # Try using integer sizes for smoother drawing(?)
            pen.setWidth(max(1, int(round(2.0 / self.scale))))
            painter.setPen(pen)

            line_path = QPainterPath()
            vrtx_path = QPainterPath()

            if self.shape_type == 'rectangle':
                assert len(self.points) in [1, 2]
                if len(self.points) == 2:
                    rectangle = self.getRectFromLine(*self.points)
                    line_path.addRect(rectangle)
                for i in range(len(self.points)):
                    self.drawVertex(vrtx_path, i)
            elif self.shape_type == "circle":
                assert len(self.points) in [1, 2]
                if len(self.points) == 2:
                    rectangle = self.getCircleRectFromLine(self.points)
                    line_path.addEllipse(rectangle)
                for i in range(len(self.points)):
                    self.drawVertex(vrtx_path, i)
            elif self.shape_type == "linestrip":
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

            painter.drawPath(line_path)
            painter.drawPath(vrtx_path)
            painter.fillPath(vrtx_path, self.vertex_fill_color)
            if self.fill:
                color = self.select_fill_color \
                    if self.selected else self.fill_color
                painter.fillPath(line_path, color)

    def drawVertex(self, path, i):
        d = self.point_size / self.scale
        shape = self.point_type
        point = self.points[i]
        if i == self._highlightIndex:
            size, shape = self._highlightSettings[self._highlightMode]
            d *= size
        if self._highlightIndex is not None:
            self.vertex_fill_color = self.hvertex_fill_color
        else:
            self.vertex_fill_color = Shape.vertex_fill_color
        if shape == self.P_SQUARE:
            path.addRect(point.x() - d / 2, point.y() - d / 2, d, d)
        elif shape == self.P_ROUND:
            path.addEllipse(point, d / 2.0, d / 2.0)
        else:
            assert False, "unsupported vertex shape"

    def nearestVertex(self, point, epsilon):
        min_distance = float('inf')
        min_i = None
        for i, p in enumerate(self.points):
            dist = distance(p - point)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                min_i = i
        return min_i

    def nearestEdge(self, point, epsilon):
        min_distance = float('inf')
        post_i = None
        for i in range(len(self.points)):
            line = [self.points[i - 1], self.points[i]]
            dist = distancetoline(point, line)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                post_i = i
        return post_i

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

    def highlightClear(self):
        self._highlightIndex = None

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
        self.epsilon = kwargs.pop('epsilon', 11.0)
        super(Canvas, self).__init__(*args, **kwargs)
        # Initialise local state.
        self.mode = self.EDIT
        self.shapes = []
        self.shapesBackups = []
        self.current = None
        self.selectedShape = None  # save the selected shape here
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
        self.visible = {}
        self._hideBackround = False
        self.hideBackround = False
        self.hShape = None
        self.hVertex = None
        self.hEdge = None
        self.movingShape = False
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
           'line', 'point', 'linestrip']:
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

    def mouseMoveEvent(self, ev):
        """Update line with last point and current coordinates."""
        try:
            if QT5:
                pos = self.transformPos(ev.pos())
            else:
                pos = self.transformPos(ev.posF())
        except AttributeError:
            return

        self.prevMovePoint = pos
        self.restoreCursor()

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
            if self.createMode in ['polygon', 'linestrip']:
                self.line[0] = self.current[-1]
                self.line[1] = pos
            elif self.createMode == 'rectangle':
                self.line.points = [self.current[0], pos]
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
        self.setToolTip("Image")
        for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index = shape.nearestVertex(pos, self.epsilon)
            index_edge = shape.nearestEdge(pos, self.epsilon)
            if index is not None:
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.hVertex = index
                self.hShape = shape
                self.hEdge = index_edge
                shape.highlightVertex(index, shape.MOVE_VERTEX)
#                self.overrideCursor(CURSOR_POINT)
                self.setToolTip("Click & drag to move point")
                self.setStatusTip(self.toolTip())
                self.update()
                break
#            elif shape.containsPoint(pos):
#                if self.selectedVertex():
#                    self.hShape.highlightClear()
#                self.hVertex = None
#                self.hShape = shape
#                self.hEdge = index_edge
#                self.setToolTip(
#                    "Click & drag to move shape '%s'" % shape.label)
#                self.setStatusTip(self.toolTip())
#                self.overrideCursor(CURSOR_GRAB)
#                self.update()
#                break
        else:  # Nothing found, clear highlights, reset state.
            if self.hShape:
                self.hShape.highlightClear()
                self.update()
            self.hVertex, self.hShape, self.hEdge = None, None, None
        self.edgeSelected.emit(self.hEdge is not None)

    def addPointToEdge(self):
        if (self.hShape is None and
                self.hEdge is None and
                self.prevMovePoint is None):
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
                if self.current:
                    # Add point to existing shape.
                    if self.createMode == 'polygon':
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if self.current.isClosed():
                            self.finalise()
                    elif self.createMode in ['rectangle', 'circle', 'line']:
                        assert len(self.current.points) == 1
                        self.current.points = self.line.points
                        self.finalise()
                    elif self.createMode == 'linestrip':
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if int(ev.modifiers()) == Qt.ControlModifier:
                            self.finalise()
                elif not self.outOfPixmap(pos):
                    # Create new shape.
                    self.current = Shape(shape_type=self.createMode)
                    self.current.addPoint(pos)
                    if self.createMode == 'point':
                        self.finalise()
                    else:
                        if self.createMode == 'circle':
                            self.current.shape_type = 'circle'
                        self.line.points = [pos, pos]
                        self.setHiding()
                        self.drawingPolygon.emit(True)
                        self.update()
            else:
                self.selectShapePoint(pos)
                self.prevPoint = pos
                self.repaint()
        elif ev.button() == Qt.RightButton and self.editing():
            self.selectShapePoint(pos)
            self.prevPoint = pos
            self.repaint()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.RightButton:
            menu = self.menus[bool(self.selectedShapeCopy)]
            self.restoreCursor()
            if not menu.exec_(self.mapToGlobal(ev.pos()))\
               and self.selectedShapeCopy:
                # Cancel the move by deleting the shadow copy.
                self.selectedShapeCopy = None
                self.repaint()
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
            return
        for shape in reversed(self.shapes):
            if self.isVisible(shape) and shape.containsPoint(point):
                shape.selected = True
                self.selectedShape = shape
                self.calculateOffsets(shape, point)
                self.setHiding()
                self.selectionChanged.emit(True)
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
        point = shape[index]
        if self.outOfPixmap(pos):
            pos = self.intersectionPoint(point, pos)
        shape.moveVertexBy(index, pos - point)

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

        p.end()

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
        self.shapes.append(self.current)
        self.storeShapes()
        self.current = None
        self.setHiding(False)
        self.newShape.emit()
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
        if self.createMode in ['polygon', 'linestrip']:
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


__appname__ = 'mylabelingapp'

class MainWindow(QMainWindow):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))

    def __init__(self, defaultFilename=None, defaultPrefdefClassFile=None, defaultSaveDir=None):
        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)
        self.showNormal()
        self.setGeometry(30 ,30 ,1300 ,695)
        self.setWindowIcon(QIcon('icons/appicon.png'))
        
        self.zoomWidget = ZoomWidget()
        
        self.canvas = Canvas(parent=self)
        self.canvas.zoomRequest.connect(self.zoomRequest)
        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.shapeMoved)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)
        
        self.zoomWidget.valueChanged.connect(self.paintCanvas)
        
        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal: scroll.horizontalScrollBar()
        }
        
        self.zoomMode = self.MANUAL_ZOOM
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }
        self.scrollArea = scroll
        self.canvas.scrollRequest.connect(self.scrollRequest)
        
        self.setCentralWidget(scroll)
        
        self.image = QImage()
        self.video = None
        self.videobuffer = []
        self.frameNum = None
        self.total_frames = None
        self.Play_status = False
        self._noSelectionSlot = False
        self.filePath = None
        self.csvFilename = None
        self.csvdata = []
        self.shouldItrack = True
        self.canclose = True
        self.opened = not self.canclose
        self.previous_Scroll_pos = self.scrollBars[Qt.Horizontal].value(),self.scrollBars[Qt.Vertical].value()
        
        action = functools.partial(newAction,self)
        
        createPolygonMode = action(
            'Create Polygons',
            lambda: self.toggleDrawMode(False, createMode='polygon'),
            'Ctrl+P',
            'polygon',
            'Start drawing polygons',
            enabled=False,
        )
        createRectangleMode = action(
            'Create Rectangle',
            lambda: self.toggleDrawMode(False, createMode='rectangle'),
            'Ctrl+R',
            'rectangle',
            'Start drawing rectangles',
            enabled=False,
        )
        createCircleMode = action(
            'Create Circle',
            lambda: self.toggleDrawMode(False, createMode='circle'),
            'Ctrl+C',
            'circle',
            'Start drawing circles',
            enabled=False,
        )
        createLineMode = action(
            'Create Line',
            lambda: self.toggleDrawMode(False, createMode='line'),
            'Ctrl+L',
            'rectangle',
            'Start drawing lines',
            enabled=False,
        )
        createPointMode = action(
            'Create Point',
            lambda: self.toggleDrawMode(False, createMode='point'),
            'P',
            'dot',
            'Start drawing points',
            enabled=False,
        )
        createLineStripMode = action(
            'Create LineStrip',
            lambda: self.toggleDrawMode(False, createMode='linestrip'),
            'L',
            'polyline',
            'Start drawing linestrip. Ctrl+LeftClick ends creation.',
            enabled=False,
        )
        addPoint = action('Add Point to Edge', self.canvas.addPointToEdge,
                          None, 'edit', 'Add point to the nearest edge',
                          enabled=False)
        
        editMode = action('Edit Rectangle', self.setEditMode,'Ctrl+E', 'edit',
                  'Move and Edit Rectangle',enabled=False,)

        shapeLineColor = action(
            'Shape &Line Color', self.chshapeLineColor, icon='color-line',
            tip='Change the line color for this specific shape', enabled=False)
        shapeFillColor = action(
            'Shape &Fill Color', self.chshapeFillColor, icon='fill-color',
            tip='Change the fill color for this specific shape', enabled=False)
        
        Quit = action('Quit', self.close, 'Ctrl+Q', 'quit',
                      'Quit application')
        Openfolder  = action(
                'Open Folder',
                lambda: self.openFile(filetype='dir'), 
                'Ctrl+O', 'open-folder',
                'Open Folder'
                )
        Openfile  = action(
                'Open File', 
                lambda: self.openFile(filetype='file'), 
                'Ctrl+O', 'open-files',
                'Open File'
                )
        Close  = action('Close File', self.closeFile, 'Ctrl+Shift+W', 'close',
                      'Close File')
        
        delete = action("Delete Rectangle", self.deleteSelectedShape,'Delete', 'delete', 
                        "Delete selected Rectangle",)
        
        undo = action("Undo Deletion", self.undoDeletetion,'Ctrl+Z', 'undo', 
                        "Undo Deletion",)
        menu=(
            createPolygonMode,
            createRectangleMode,
            createCircleMode,
            createLineMode,
            createPointMode,
            createLineStripMode,
            editMode,
            delete,
            shapeLineColor,
            shapeFillColor,
            undo,
            addPoint,
        )
        onLoadActive=(
#            close,
            createPolygonMode,
            createRectangleMode,
            createCircleMode,
            createLineMode,
            createPointMode,
            createLineStripMode,
            editMode,
        )
        self.actions = struct(createPolygonMode=createPolygonMode,createCircleMode=createCircleMode,
                              createRectangleMode=createRectangleMode,addPoint=addPoint,
                              createLineMode=createLineMode, createPointMode=createPointMode,
                              createLineStripMode=createLineStripMode,menu=menu,onLoadActive=onLoadActive,
                              shapeLineColor=shapeLineColor,shapeFillColor=shapeFillColor,
                              Quit=Quit, editMode=editMode, delete=delete, Openfile=Openfile,
                              Openfolder=Openfolder,undo=undo, Close=Close)
        
        addActions(self.canvas.menus[0], self.actions.menu)
        self.canvas.edgeSelected.connect(self.actions.addPoint.setEnabled)
#        self.actions.AiMode.setEnabled(False)
        
        button = functools.partial(newButton,self)
        self.fitWindow_button = button('&Fit-To-Canvas', self.setFitWindow,'Tab', 'expand',
                           'Zoom to window size',style=Qt.ToolButtonTextBesideIcon,checkable=True,)
        
        self.play_button = button('&Play', self.togglePlayMode,'Space', 'play',
                           'Start playing video',stylesheet="border-radius: 4px;",IconSize=QSize(65,65))
        
        self.previous_button = button('&Previous', self.loadPreviousframe,'A', 'previous',
                           'Go to previous frame',stylesheet="border-radius: 4px;", IconSize=QSize(45,45))
        
        self.next_button = button('&Next', self.loadNextframe,'D', 'next',
                           'Go to next frame',stylesheet="border-radius: 4px;",IconSize=QSize(45,45))
        
        self.usePrevious_botton = button('&Use Previous', self.loadPreviousFrameShapes,'I', 'use-previous',
                           'Use previous frame shapes',style=Qt.ToolButtonTextBesideIcon)
                
        full_screen = button('Go full screen', self.toggleFullscreen,'Ctrl+Tab' ,'full-screen',
                             'Go full screen',)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(self.actions.Openfile)
        fileMenu.addAction(self.actions.Openfolder)
        fileMenu.addAction(self.actions.Close)
        helpmenu = menubar.addMenu('&Help')
        
        toolbar = self.addToolBar('Quick Access')
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        toolbar.addAction(self.actions.Openfile)
        toolbar.addAction(self.actions.Openfolder)
        toolbar.addAction(self.actions.Quit)
        
        self.labelCoordinates = QLabel('')
        self.statusBar().addPermanentWidget(self.labelCoordinates)
        self.statusBar().addPermanentWidget(self.zoomWidget.zoomlabel)
        self.statusBar().addPermanentWidget(self.zoomWidget)
        self.statusBar().addPermanentWidget(self.fitWindow_button)
        self.statusBar().addPermanentWidget(full_screen)
        
        self.zoomWidget.setEnabled(False)
        self.fitWindow_button.setEnabled(False)
        
    def closeFile(self):
        pass
    
    def openFile(self,filetype='file'):
        try:
            if filetype == 'file':
                filename = QFileDialog.getOpenFileName(self, '%s - Open file' % __appname__, '',
                    'Video Files (*.avi;*.mp4);; Image Files (*.jpg; *.png; *.jpeg; *.bmp)')[0]
                if filename and not os.path.isdir(filename):
                    self.queueEvent(functools.partial(self.loadFile, filename or ""))
                    
            elif filetype == 'dir':
                targetDirPath = str(QFileDialog.getExistingDirectory(
                        self, '%s - Open Directory' % __appname__,'',
                        QFileDialog.ShowDirsOnly |
                        QFileDialog.DontResolveSymlinks))
                self.loadDirImages(targetDirPath)
                
        except IOError or AttributeError:
            QMessageBox.warning(self, "Warning", "No file selected")

    def closeEvent(self, event):
        self.reply = QMessageBox.question(self, 'Confirm Exit',
            "Are you sure to Quit Application ?", QMessageBox.Yes | 
            QMessageBox.No, QMessageBox.No)

        if self.reply == QMessageBox.Yes:               
            event.accept()
        else:
            event.ignore()      
       
    def queueEvent(self, function):
        QTimer.singleShot(0, function)
            
    def convertToPixmap(self, imageData):
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
        
        return None
    
    def readVideo(self,FilePath):
        return cv2.VideoCapture(FilePath)
    
    def loadPreviousFrameShapes(self):
        print("use previous doesnot working")
        
    def loadFile(self,FilePath):
        self.filePath = FilePath
        self.resetState()
        self.canvas.setEnabled(False)
        if ('.avi' in FilePath or '.mp4' in FilePath) and self.video is None:
            self.video = self.readVideo(FilePath)
            self.totalframes = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            while self.video.isOpened():
                ret, frame = self.video.read()
                if ret is True:
                    self.videobuffer.append(frame)
                else:
                    break
            self.video.release()
            cv2.destroyAllWindows()
            self.frameNum = 0
            self.image = self.convertToPixmap(self.videobuffer[self.frameNum])
        elif '.png' in FilePath or '.jpg' in FilePath or '.jpeg' in FilePath or '.bmp' in FilePath:
    #        image = self.convertToPixmap(read(FilePath, None))
            image = self.convertToPixmap(cv2.imread(FilePath))
            if image.isNull():
                print("image not loaded")
                return False
            self.image = image
        else:
            print("image not supported")
            return False
        self.loadPixmapToCanvas()
        self.toggleActions(True)
    def loadDirImages(self, dirpath, pattern=None, load=True):
        pass
    
    def loadPixmapToCanvas(self):        
        self.canvas.loadPixmap(QPixmap.fromImage(self.image))
        self.canvas.setEnabled(True)
        self.zoomWidget.setEnabled(True)
        self.fitWindow_button.setEnabled(True)
        self.adjustScale()
        self.paintCanvas()
                
    def loadNextframe(self):
#        print("next")
        self.previous_Scroll_pos = self.scrollBars[Qt.Horizontal].value(),self.scrollBars[Qt.Vertical].value()
        self.canvas.setEnabled(False)
        if not len(self.videobuffer) == 0 and (self.totalframes <= len(self.videobuffer)):
            if self.frameNum+1 >= len(self.videobuffer):
                return False
            self.frameNum += 1
            self.image = self.convertToPixmap(self.videobuffer[self.frameNum])
            self.loadPixmapToCanvas()
            self.scrollBars[Qt.Horizontal].setValue(self.previous_Scroll_pos[0])
            self.scrollBars[Qt.Vertical].setValue(self.previous_Scroll_pos[1])
            return True
        return False

    def loadPreviousframe(self):
#        print("previous")
        self.previous_Scroll_pos = self.scrollBars[Qt.Horizontal].value(),self.scrollBars[Qt.Vertical].value()
        self.canvas.setEnabled(False)
        if not len(self.videobuffer) == 0 and (self.totalframes <= len(self.videobuffer)):
            if self.frameNum-1 < 0:
                return False
            self.frameNum -= 1
            self.image = self.convertToPixmap(self.videobuffer[self.frameNum])
            self.loadPixmapToCanvas()
            self.scrollBars[Qt.Horizontal].setValue(self.previous_Scroll_pos[0])
            self.scrollBars[Qt.Vertical].setValue(self.previous_Scroll_pos[1])
            return True
        return False

    def toggleDrawingSensitive(self, drawing=True):
        """In the middle of drawing, toggling between modes should be disabled."""
        
        if not drawing:
            self.canvas.setEditing(True)
            self.actions.editMode.setEnabled(not drawing)
    #        self.actions.undoLastPoint.setEnabled(drawing)
            self.actions.undo.setEnabled(not drawing)
            self.actions.delete.setEnabled(not drawing)
            self.actions.createPolygonMode.setEnabled(True)
            self.actions.createRectangleMode.setEnabled(True)
            self.actions.createCircleMode.setEnabled(True)
            self.actions.createLineMode.setEnabled(True)
            self.actions.createPointMode.setEnabled(True)
            self.actions.createLineStripMode.setEnabled(True)

    def toggleDrawMode(self, edit=True, createMode='polygon'):
        self.canvas.setEditing(edit)
        self.canvas.createMode = createMode
        if edit:
            self.actions.createPolygonMode.setEnabled(True)
            self.actions.createRectangleMode.setEnabled(True)
            self.actions.createCircleMode.setEnabled(True)
            self.actions.createLineMode.setEnabled(True)
            self.actions.createPointMode.setEnabled(True)
            self.actions.createLineStripMode.setEnabled(True)
        else:
            if createMode == 'polygon':
                self.actions.createPolygonMode.setEnabled(False)
                self.actions.createRectangleMode.setEnabled(True)
                self.actions.createCircleMode.setEnabled(True)
                self.actions.createLineMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(True)
                self.actions.createLineStripMode.setEnabled(True)
            elif createMode == 'rectangle':
                self.actions.createPolygonMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(False)
                self.actions.createCircleMode.setEnabled(True)
                self.actions.createLineMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(True)
                self.actions.createLineStripMode.setEnabled(True)
            elif createMode == 'line':
                self.actions.createPolygonMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(True)
                self.actions.createCircleMode.setEnabled(True)
                self.actions.createLineMode.setEnabled(False)
                self.actions.createPointMode.setEnabled(True)
                self.actions.createLineStripMode.setEnabled(True)
            elif createMode == 'point':
                self.actions.createPolygonMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(True)
                self.actions.createCircleMode.setEnabled(True)
                self.actions.createLineMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(False)
                self.actions.createLineStripMode.setEnabled(True)
            elif createMode == "circle":
                self.actions.createPolygonMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(True)
                self.actions.createCircleMode.setEnabled(False)
                self.actions.createLineMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(True)
                self.actions.createLineStripMode.setEnabled(True)
            elif createMode == "linestrip":
                self.actions.createPolygonMode.setEnabled(True)
                self.actions.createRectangleMode.setEnabled(True)
                self.actions.createCircleMode.setEnabled(True)
                self.actions.createLineMode.setEnabled(True)
                self.actions.createPointMode.setEnabled(True)
                self.actions.createLineStripMode.setEnabled(False)
            else:
                raise ValueError('Unsupported createMode: %s' % createMode)
        self.actions.editMode.setEnabled(not edit)

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

    def togglePlayMode(self):
        if self.video.isOpened():
            self.Play_status = not self.Play_status
            
    def toggleFullscreen(self):
        if self.isFullScreen() is True:
            self.showNormal()
            self.showMaximized()
        else:
            self.showFullScreen()

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
#        for z in self.actions.zoomActions:
#            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def editShapeProperty(self):
#        print("property")
        pass
    
    # React to canvas signals.
    @Slot(bool)
    def shapeSelectionChanged(self, selected=False):
        self.actions.delete.setEnabled(selected)
#        self.actions.copy.setEnabled(selected)
#        self.actions.edit.setEnabled(selected)
        self.actions.shapeLineColor.setEnabled(selected)
        self.actions.shapeFillColor.setEnabled(selected)
        
    def ShapePropertyChanged(self):
#        print("shape property changed")
        pass

    def currentItem(self):
        pass    
    
    @Slot()
    def newShape(self):
        pass
#            print("hhh")
        
    def shapeMoved(self):
        pass
        
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
        
    def resetModes(self):
        self.canvas.setEditing(True)
        self.actions.createMode.setEnabled(True)
        self.actions.editMode.setEnabled(False)
        self.actions.AiMode.setEnabled(True)
        
    def resizeEvent(self, event):
        try:
            if self.canvas and not self.image.isNull()\
               and self.zoomMode != self.MANUAL_ZOOM:
                self.adjustScale()
        except AttributeError:
            pass
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
#        assert not self.image.isNull(), "cannot paint null image"
        if not self.image.isNull():
            self.canvas.scale = 0.01 * self.zoomWidget.value()
            self.canvas.adjustSize()
            self.canvas.update()
    
    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        self.zoomWidget.setValue(int(100 * value))
        
    def scaleFitWindow(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width()
        h2 = self.canvas.pixmap.height()
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()
    
    def scrollRequest(self, delta, orientation):
        units = - delta / (8 * 15)
        bar = self.scrollBars[orientation]
        bar.setValue(bar.value() + bar.singleStep() * units)

    def setZoom(self, value):
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)

    def addZoom(self, increment=10):
        self.setZoom(self.zoomWidget.value() + increment)

    @Slot(int, QPoint)
    def zoomRequest(self, delta,pos):
        canvas_width_old = self.canvas.width()
        units = delta * 0.1
        self.addZoom(units)

        canvas_width_new = self.canvas.width()
        if canvas_width_old != canvas_width_new:
            canvas_scale_factor = canvas_width_new / canvas_width_old

            x_shift = round(pos.x() * canvas_scale_factor) - pos.x()
            y_shift = round(pos.y() * canvas_scale_factor) - pos.y()
            self.scrollBars[Qt.Horizontal].setValue(
                self.scrollBars[Qt.Horizontal].value() + x_shift)
            self.scrollBars[Qt.Vertical].setValue(
                self.scrollBars[Qt.Vertical].value() + y_shift)

    def setFitWindow(self, value=True):
#        if value:
#            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
#        if value:
#            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

class ZoomWidget(QSpinBox):
    def __init__(self, value=100):
        super(ZoomWidget, self).__init__()
#        self.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.zoomlabel = QLabel('<b>Zoom:<\b> ')
        self.setRange(1, 500)
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
    
class struct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


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

def newIcon(icon):
#    here = os.path.dirname(os.path.abspath(__file__))
#    icons_dir = os.path.join(here, '../icons')
    return QIcon(r'icons/%s.png' % icon)

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
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())  