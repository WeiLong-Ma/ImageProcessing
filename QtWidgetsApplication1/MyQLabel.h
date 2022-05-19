#pragma once

#include <QObject>
#include <QLABEL>
#include <QtWidgets/qwidget.h>
#include <QMouseEvent>
class MyQLabel : public QLabel
{
	Q_OBJECT
public:
	explicit MyQLabel(QWidget *parent = nullptr);
	~MyQLabel();
	void mousePressEvent(QMouseEvent* e);
	void mouseDoubleClickEvent(QMouseEvent* e);
	int x, y,px,py;
signals:
	void MouseDP();
	void MousePos();
	void MousePPos();
};
