#include "qtwidgetsapplication1.h"
#include <QMessageBox>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/imgproc/types_c.h>
#pragma execution_character_set("utf-8")//QT中文亂碼

using namespace cv;


QImage ori_image;
//QMessageBox::warning(NULL, "my messagebox", "hello world");//測試用
QtWidgetsApplication1::QtWidgetsApplication1(QWidget *parent)
    : QWidget(parent)
{
    ui.setupUi(this);
}
// cv::Mat轉換成QImage
QImage cvMat2QImage(const Mat& mat)
{
    if (mat.type() == CV_8UC1)                          // 單通道
    {
        QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);
        image.setColorCount(256);                       // 灰度級數256
        for (int i = 0; i < 256; i)
        {
            image.setColor(i, qRgb(i, i, i));
        }
        uchar* pSrc = mat.data;                         // 複製mat資料
        for (int row = 0; row < mat.rows; row)
        {
            uchar* pDest = image.scanLine(row);
            memcpy(pDest, pSrc, mat.cols);
            pSrc = mat.data;
        }
        return image;
    }
    else if (mat.type() == CV_8UC3)                     // 3通道
    {
        const uchar* pSrc = (const uchar*)mat.data;     // 複製畫素
        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);    // R, G, B 對應 0,1,2
        return image.rgbSwapped();                      // rgbSwapped是為了顯示效果色彩好一些。
    }
    else if (mat.type() == CV_8UC4)                     // 4通道
    {
        const uchar* pSrc = (const uchar*)mat.data;     // 複製畫素
        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);        // B,G,R,A 對應 0,1,2,3
        return image.copy();
    }
    else
    {
        return QImage();
    }
}
// QImage轉換成cv::Mat
Mat QImage2cvMat(QImage image)
{
    Mat mat;
    switch (image.format())
    {
    case QImage::Format_ARGB32:
    case QImage::Format_RGB32:
    case QImage::Format_ARGB32_Premultiplied:
        mat = Mat(image.height(), image.width(), CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
        break;
    case QImage::Format_RGB888:
        mat = Mat(image.height(), image.width(), CV_8UC3, (void*)image.constBits(), image.bytesPerLine());
        cv::cvtColor(mat, mat, CV_BGR2RGB);
        break;
    case QImage::Format_Indexed8:
        mat = Mat(image.height(), image.width(), CV_8UC1, (void*)image.constBits(), image.bytesPerLine());
        break;
    }
    return mat;
}


void QtWidgetsApplication1::on_OpenFileButton_Clicked() { //qt右下信號編輯
    QString filename = QFileDialog::getOpenFileName(this, tr("選擇"), "", tr("Images(*.png *.jpg *.jpeg *.gif)"));
    if (filename != 0) {
        QImage image;
        bool valid = image.load(filename);
        
        if (valid) {
            image = image.scaledToWidth(ui.image1->width(), Qt::SmoothTransformation);
            /*image = image.scaledToHeight(ui.image1->height(), Qt::SmoothTransformation);
            ui.image1->setScaledContents(true); //尺寸跟label一樣*/
            ui.image1->setPixmap(QPixmap::fromImage(image));
            
            //imshow("123",QImage2cvMat(ori_image));
        }
        else {
            QMessageBox::warning(NULL, "訊息方塊", "錯誤");
        }
    }
}
void QtWidgetsApplication1::imageROI() {
    int x1, x2, y1, y2;
    QPixmap test;
    x1 = ui.imageX1->toPlainText().toInt();
    x2 = ui.imageX2->toPlainText().toInt();
    y1 = ui.imageY1->toPlainText().toInt();
    y2 = ui.imageY2->toPlainText().toInt();
    if (x1 != NULL && x2 != NULL && y1 != NULL && y2 != NULL) {
        //QImage cimage = ori_image.copy(x1, y1, x2, y2);
        //ui.image1->setPixmap(QPixmap::fromImage(cimage)); //QT影像分割
        Mat Rimage = QImage2cvMat(ui.image1->pixmap()->toImage());
        Rimage = Rimage(Range(x1, x2), Range(y1, y2));
        imshow("test",Rimage);
    }
    else {
        QMessageBox::warning(NULL, "訊息方塊", "請輸入數值");
    }
}
void QtWidgetsApplication1::histimage() {
    if (ui.image1->pixmap() != NULL) {
        Mat Ori_image = QImage2cvMat(ui.image1->pixmap()->toImage());
        Mat Gray_image;
        cvtColor(Ori_image, Gray_image, COLOR_BGR2GRAY);

        Mat dstHist;
        int dims = 1;
        float hranges[] = { 0, 256 };
        const float* ranges[] = {hranges};
        int size = 256;
        int channels = 0;

        //計算影像的直方圖
        calcHist(&Gray_image, 1, &channels, Mat(), dstHist, dims, &size, ranges);

        Mat dstImage(size, size, CV_8U, Scalar(0));
        //獲得最小值最大值
        double minValue = 0;
        double maxValue = 0;
        minMaxLoc(dstHist, &minValue, &maxValue, 0, 0);  //在cv中用的是cvGetMinMaxHistValue

        //繪製直方圖
        //saturate_cast函數的作用即是：當運算完之後，最小值0最大值為255。
        int hpt = saturate_cast<int>(0.9 * size);
        for (int i = 0; i < 256; i++)
        {
            float binValue = dstHist.at<float>(i);//hist類別為float 
            //拉伸到0-max
            int realValue = saturate_cast<int>(binValue * hpt / maxValue);
            line(dstImage, Point(i, size - 1), Point(i, size - realValue), Scalar(255));
        }
        imshow("直方圖", dstImage);
    }
    else {
        QMessageBox::warning(NULL, "訊息方塊", "請載入圖片");
    }
}
void QtWidgetsApplication1::Thresholding() {
    if (ui.image1->pixmap() != NULL) {
        Mat Ori_image = QImage2cvMat(ui.image1->pixmap()->toImage());
        Mat Gray_image, Timage;
        int tvalue = ui.ThresholdingSlider->value();
        cvtColor(Ori_image, Gray_image, COLOR_BGR2GRAY);
        threshold(Gray_image, Timage, tvalue, 255, THRESH_BINARY);
        imshow("影像二值化", Timage);
    }
    else {
        QMessageBox::warning(NULL, "訊息方塊", "請載入圖片");
    }
}
void QtWidgetsApplication1::HEqualization() {
    if (ui.image1->pixmap() != NULL) {
        Mat Ori_image = QImage2cvMat(ui.image1->pixmap()->toImage());
        Mat Gray_image, Eimage;
        cvtColor(Ori_image, Gray_image, COLOR_BGR2GRAY);
        equalizeHist(Gray_image,Eimage);
        imshow("影像直方圖等化", Eimage);

        Mat dstHist;
        int dims = 1;
        float hranges[] = { 0, 256 };
        const float* ranges[] = { hranges };
        int size = 256;
        int channels = 0;

        //計算影像的直方圖
        calcHist(&Eimage, 1, &channels, Mat(), dstHist, dims, &size, ranges);

        Mat dstImage(size, size, CV_8U, Scalar(0));
        //獲得最小值最大值
        double minValue = 0;
        double maxValue = 0;
        minMaxLoc(dstHist, &minValue, &maxValue, 0, 0);  //在cv中用的是cvGetMinMaxHistValue

        //繪製直方圖
        //saturate_cast函數的作用即是：當運算完之後，最小值0最大值為255。
        int hpt = saturate_cast<int>(0.9 * size);
        for (int i = 0; i < 256; i++)
        {
            float binValue = dstHist.at<float>(i);//hist類別為float 
            //拉伸到0-max
            int realValue = saturate_cast<int>(binValue * hpt / maxValue);
            line(dstImage, Point(i, size - 1), Point(i, size - realValue), Scalar(255));
        }
        imshow("等化後直方圖", dstImage);
    }
    else {
        QMessageBox::warning(NULL, "訊息方塊", "請載入圖片");
    }
}

void QtWidgetsApplication1::ToGray() {
    if (ui.image1->pixmap() != NULL) {
        Mat Ori_image = QImage2cvMat(ui.image1->pixmap()->toImage());
        Mat Gray_image;
        cvtColor(Ori_image, Gray_image, COLOR_BGR2GRAY);
        imshow("Gray", Gray_image);
    }
    else {
        QMessageBox::warning(NULL, "訊息方塊", "請載入圖片");
    }
}

void QtWidgetsApplication1::ToHSV() {
    if (ui.image1->pixmap() != NULL) {
        Mat Ori_image = QImage2cvMat(ui.image1->pixmap()->toImage());
        Mat HSV_image;
        cvtColor(Ori_image, HSV_image, COLOR_BGR2HSV);
        imshow("HSV", HSV_image);
    }
    else {
        QMessageBox::warning(NULL, "訊息方塊", "請載入圖片");
    }
}
