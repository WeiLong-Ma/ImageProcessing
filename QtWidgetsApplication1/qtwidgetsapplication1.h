#pragma once

#include <QtWidgets/QWidget>
#include <QtWidgets/QFileDialog>
#include "ui_qtwidgetsapplication1.h"

class QtWidgetsApplication1 : public QWidget
{
    Q_OBJECT

public:
    QtWidgetsApplication1(QWidget *parent = Q_NULLPTR);

private:
    Ui::QtWidgetsApplication1Class ui;

private slots:
    void on_OpenFileButton_Clicked();
    void imageROI();
    void histimage();
    void Thresholding();
    void HEqualization();
    void ToGray();
    void ToHSV();
};
