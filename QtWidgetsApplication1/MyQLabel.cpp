#include "MyQLabel.h"
#include <QMessageBox>

MyQLabel::MyQLabel(QWidget *parent)
	: QLabel(parent)
{
}

MyQLabel::~MyQLabel()
{
}

void MyQLabel::mousePressEvent(QMouseEvent* e)
{
	if (e->buttons() == Qt::LeftButton) {
		this->x = e->x();
		this->y = e->y();
		emit MousePos();
	}
}
void MyQLabel::mouseDoubleClickEvent(QMouseEvent* e)
{
	if (e->buttons() == Qt::LeftButton) {
		this->px = e->x();
		this->py = e->y();
		emit MousePPos();
	}
}

