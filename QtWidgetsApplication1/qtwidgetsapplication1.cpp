#include "qtwidgetsapplication1.h"
#include <QMessageBox>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/imgproc/types_c.h>

#pragma execution_character_set("utf-8")//QT中文亂碼

using namespace cv;
int PTPointX[4] = { 0,0,0,0 };
int PTPointY[4] = { 0,0,0,0 };
int Pcounter = 0;

//QMessageBox::warning(NULL, "my messagebox", "hello world");//測試用
QtWidgetsApplication1::QtWidgetsApplication1(QWidget *parent)
    : QWidget(parent)
{
    ui.setupUi(this);
    connect(ui.image1, SIGNAL(MousePos()), this, SLOT(Mouse_Pressed()));
    //連結MyQLabel.h的MousePos()與qtwid..h的Mouse_Pressed()
    connect(ui.image1, SIGNAL(MousePPos()), this, SLOT(PerspectiveTransformPoint()));
    QPixmap pixmap1("ClownGlasses.png");
    ui.decoration_image1->setPixmap(pixmap1);
    ui.decoration_image1->setScaledContents(true); 
    QPixmap pixmap2("dog.png");
    ui.decoration_image2->setPixmap(pixmap2);
    ui.decoration_image2->setScaledContents(true);
    QPixmap pixmap3("SantaClaus.png");
    ui.decoration_image3->setPixmap(pixmap3);
    ui.decoration_image3->setScaledContents(true);
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
void QtWidgetsApplication1::savefile() {
    QString filename = QFileDialog::getSaveFileName(this, tr("選擇"), "", tr("Images(*.png *.jpg *.jpeg *.gif)"));
    if (filename != 0) {
        Mat image = QImage2cvMat(ui.image1->pixmap()->toImage());
        imwrite(filename.toLocal8Bit().data(), image);
    }
}

void QtWidgetsApplication1::on_OpenFileButton_Clicked() { //qt右下信號編輯
    QString filename = QFileDialog::getOpenFileName(this, tr("選擇"), "", tr("Images(*.png *.jpg *.jpeg *.gif)"));
    if (filename != 0) {
        QImage image;
        bool valid = image.load(filename);
        
        if (valid) {
            //image = image.scaledToWidth(ui.image1->width(), Qt::SmoothTransformation);
            //image = image.scaledToHeight(ui.image1->height(), Qt::SmoothTransformation);
            //ui.image1->setScaledContents(true); //尺寸跟label一樣*/
            ui.image1->setPixmap(QPixmap::fromImage(image));
            ui.image1->resize(ui.image1->pixmap()->size());
            ui.Width->setText("寬：" + QVariant(ui.image1->width()).toString());
            ui.Height->setText("高：" + QVariant(ui.image1->height()).toString());
            //QVariant可儲存各種類型的資料 支援所有QMetaType
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
        Mat Ori_image = QImage2cvMat(ui.image1->pixmap()->toImage());
        Mat Rimage = Ori_image(Rect(x1,y1,x2-x1,y2-y1));
        imshow("ROI Image",Rimage);
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
        imshow("Histogram", dstImage);
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
        imshow("Thresholding", Timage);
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
        imshow("Histogram Equalization1", Eimage);

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
        imshow("Histogram Equalization2", dstImage);
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
void QtWidgetsApplication1::ToYCrCb() {
    if (ui.image1->pixmap() != NULL) {
        Mat Ori_image = QImage2cvMat(ui.image1->pixmap()->toImage());
        Mat HSV_image;
        cvtColor(Ori_image, HSV_image, COLOR_BGR2YCrCb);
        imshow("YCrCb", HSV_image);
    }
    else {
        QMessageBox::warning(NULL, "訊息方塊", "請載入圖片");
    }
}
void QtWidgetsApplication1::ToRGB() {
    if (ui.image1->pixmap() != NULL) {
        Mat Ori_image = QImage2cvMat(ui.image1->pixmap()->toImage());
        Mat HSV_image;
        cvtColor(Ori_image, HSV_image, COLOR_BGR2RGB);
        imshow("RGB", HSV_image);
    }
    else {
        QMessageBox::warning(NULL, "訊息方塊", "請載入圖片");
    }
}
void QtWidgetsApplication1::ToYUV() {
    if (ui.image1->pixmap() != NULL) {
        Mat Ori_image = QImage2cvMat(ui.image1->pixmap()->toImage());
        Mat HSV_image;
        cvtColor(Ori_image, HSV_image, COLOR_BGR2YUV);
        imshow("YUV", HSV_image);
    }
    else {
        QMessageBox::warning(NULL, "訊息方塊", "請載入圖片");
    }
}
void QtWidgetsApplication1::ToLab() {
    if (ui.image1->pixmap() != NULL) {
        Mat Ori_image = QImage2cvMat(ui.image1->pixmap()->toImage());
        Mat HSV_image;
        cvtColor(Ori_image, HSV_image, COLOR_BGR2Lab);
        imshow("Lab", HSV_image);
    }
    else {
        QMessageBox::warning(NULL, "訊息方塊", "請載入圖片");
    }
}
void QtWidgetsApplication1::AveragingFilter() {
    if (ui.image1->pixmap() != NULL) {
        Mat Ori_image = QImage2cvMat(ui.image1->pixmap()->toImage());
        Mat Averaging_image;
        blur(Ori_image, Averaging_image, Size(7, 7), Point(1, -1), BORDER_DEFAULT);
        imshow("AveragingImage", Averaging_image);
    }
    else {
        QMessageBox::warning(NULL, "訊息方塊", "請載入圖片");
    }
}
void QtWidgetsApplication1::GaussianFilter() {
    if (ui.image1->pixmap() != NULL) {
        Mat Ori_image = QImage2cvMat(ui.image1->pixmap()->toImage());
        Mat Gaussian_image;
        GaussianBlur(Ori_image, Gaussian_image, Size(7, 7), 1.5);
        imshow("GaussianImage", Gaussian_image);
    }
    else {
        QMessageBox::warning(NULL, "訊息方塊", "請載入圖片");
    }
}
void QtWidgetsApplication1::MedianFilter() {
    if (ui.image1->pixmap() != NULL) {
        Mat Ori_image = QImage2cvMat(ui.image1->pixmap()->toImage());
        Mat Median_image;
        medianBlur(Ori_image, Median_image, 7);
        imshow("MedianImage", Median_image);
    }
    else {
        QMessageBox::warning(NULL, "訊息方塊", "請載入圖片");
    }
}
void QtWidgetsApplication1::LaplacianFilter() {
    if (ui.image1->pixmap() != NULL) {
        Mat Ori_image = QImage2cvMat(ui.image1->pixmap()->toImage());
        Mat Laplacian_image;
        Laplacian(Ori_image, Laplacian_image, CV_16S,7);
        imshow("LaplacianImage", Laplacian_image);
    }
    else {
        QMessageBox::warning(NULL, "訊息方塊", "請載入圖片");
    }
}
void QtWidgetsApplication1::SobelFilter() {
    if (ui.image1->pixmap() != NULL) {
        Mat Ori_image = QImage2cvMat(ui.image1->pixmap()->toImage());
        Mat Sobel_image,X_image,Y_image,X,Y;
        Sobel(Ori_image, X, CV_16S, 1, 0);
        Sobel(Ori_image, Y, CV_16S, 0, 1);

        convertScaleAbs(X, X_image);
        convertScaleAbs(Y, Y_image);
        addWeighted(X_image, 0.5, Y_image, 0.5, 0, Sobel_image);

        imshow("X_direction_gradient_image", X_image);
        imshow("Y_direction_gradient_image", Y_image);
        imshow("SobelImage", Sobel_image);
    }
    else {
        QMessageBox::warning(NULL, "訊息方塊", "請載入圖片");
    }
}
void QtWidgetsApplication1::TranslationImage() {
    if (ui.image1->pixmap() != NULL) {
        Mat Ori_image = QImage2cvMat(ui.image1->pixmap()->toImage());
        Mat Translation_Image;
        int Rrow = Ori_image.rows, Rcol = Ori_image.cols;
        float X = ui.TranslationX->toPlainText().toInt();
        float Y = ui.TranslationY->toPlainText().toInt();
        float M[] = {1,0,X,0,1,Y};
        Mat translation_m = Mat(2, 3, CV_32F, M); //矩陣
        warpAffine(Ori_image, Translation_Image, translation_m, Ori_image.size());
        imshow("TranslationImage", Translation_Image);
    }
    else {
        QMessageBox::warning(NULL, "訊息方塊", "請載入圖片");
    }
}
void QtWidgetsApplication1::Rotateleft() {
    if (ui.image1->pixmap() != NULL) {
        Mat Ori_image = QImage2cvMat(ui.image1->pixmap()->toImage());
        Mat Rotate_Image;
        int Rrow = Ori_image.rows, Rcol = Ori_image.cols;
        rotate(Ori_image, Rotate_Image, ROTATE_90_CLOCKWISE);
        imshow("RotateLeftImage", Rotate_Image);
    }
    else {
        QMessageBox::warning(NULL, "訊息方塊", "請載入圖片");
    }
}
void QtWidgetsApplication1::Rotateright() {
    if (ui.image1->pixmap() != NULL) {
        Mat Ori_image = QImage2cvMat(ui.image1->pixmap()->toImage());
        Mat Rotate_Image;
        int Rrow = Ori_image.rows, Rcol = Ori_image.cols;
        rotate(Ori_image, Rotate_Image, ROTATE_90_COUNTERCLOCKWISE);
        imshow("RotateRightImage", Rotate_Image);
    }
    else {
        QMessageBox::warning(NULL, "訊息方塊", "請載入圖片");
    }
}
void QtWidgetsApplication1::AffineTransform() {
    if (ui.image1->pixmap() != NULL) {
        Mat Ori_image = QImage2cvMat(ui.image1->pixmap()->toImage());
        int Rrow = Ori_image.rows, Rcol = Ori_image.cols;
        int X1 = ui.ATx1->toPlainText().toInt();
        int X2 = ui.ATx2->toPlainText().toInt();
        int X3 = ui.ATx3->toPlainText().toInt();
        int Y1 = ui.ATy1->toPlainText().toInt();
        int Y2 = ui.ATy2->toPlainText().toInt();
        int Y3 = ui.ATy3->toPlainText().toInt();
        Mat AT_Image;
        Point2f oriP[3] = { (Point2f(0, 0)), Point2f(0, Rrow), Point2f(Rcol, 0) };
        Point2f ATP[3] ={ (Point2f(X1, Y1)), Point2f(X2, Y2), Point2f(X3, Y3) };
        Mat M = getAffineTransform(oriP, ATP);
        warpAffine(Ori_image, AT_Image, M, Ori_image.size());
        imshow("AffineTransformImage", AT_Image);
    }
    else {
        QMessageBox::warning(NULL, "訊息方塊", "請載入圖片");
    }
}
void QtWidgetsApplication1::Mouse_Pressed() {
    ui.MousePoint->setText("影像點擊座標：(" + QVariant(ui.image1->x).toString() + "," + QVariant(ui.image1->y).toString()+")");
}
void QtWidgetsApplication1::PerspectiveTransformPoint() {
    PTPointX[Pcounter] = ui.image1->px;
    PTPointY[Pcounter] = ui.image1->py;
    Pcounter++;
    if (Pcounter > 3)
        Pcounter = 0;
    ui.PTpoint1->setText("X1："+ QVariant(PTPointX[0]).toString() + "｜X2：" + QVariant(PTPointX[1]).toString() + 
        +"｜X3：" + QVariant(PTPointX[2]).toString() + "｜X4：" + QVariant(PTPointX[3]).toString());
    ui.PTpoint2->setText("Y1：" + QVariant(PTPointY[0]).toString() + "｜Y2：" + QVariant(PTPointY[1]).toString() +
        +"｜Y3：" + QVariant(PTPointY[2]).toString() + "｜Y4：" + QVariant(PTPointY[3]).toString());
}
void QtWidgetsApplication1::PerspectiveTransform() {
    if (ui.image1->pixmap() != NULL) {
        Mat Ori_image = QImage2cvMat(ui.image1->pixmap()->toImage());
        int Rrow = Ori_image.rows, Rcol = Ori_image.cols;
        Mat PT_Image;
        Point2f oriP[4] = { (Point2f(0, 0)), Point2f(0, Rrow), Point2f(Rcol, 0),Point2f(Rcol,Rrow)};
        Point2f PTP[4] = { (Point2f(PTPointX[0], PTPointY[0])), Point2f(PTPointX[1], PTPointY[1]),
            Point2f(PTPointX[2], PTPointY[2]),Point2f(PTPointX[3], PTPointY[3]) };
        Mat M = getPerspectiveTransform(PTP, oriP); //內部值互換可以歪轉正
        warpPerspective(Ori_image, PT_Image, M, Ori_image.size());
        imshow("AffineTransformImage", PT_Image);
    }
    else {
        QMessageBox::warning(NULL, "訊息方塊", "請載入圖片");
    }
}

bool addphoto(cv::Mat& dst, cv::Mat& src,
    double scale = 1.0, //整體透明度
    double size = 1.0,//圖片縮放比例
    double angle = 0,//圖片旋轉角度
    cv::Point location = cv::Point(0, 0)//圖片位置
)
{
    /*if (dst.channels() != 3 || src.channels() != 4 || location.x > dst.cols || location.y > dst.cols)
    {
        return false;
    }*/


    cv::Mat small_size = src.clone();

    if (size != 1 || angle != 0) {
        int width = src.cols > (dst.cols - location.x) ? (dst.cols - location.x) : src.cols;
        int length = src.rows > (dst.rows - location.y) ? (dst.rows - location.y) : src.rows;
        cv::Mat rotation = cv::getRotationMatrix2D(cv::Point2f(length / 2, width / 2), angle, size);
        cv::warpAffine(small_size, small_size, rotation, cv::Size(width, length));
    }
    //imshow("test", small_size);
    //std::cout << small_size.cols << " " << small_size.rows << std::endl;
    cv::Mat dst_part(dst, cv::Rect(location.x, location.y, small_size.cols, small_size.rows));

    std::vector<cv::Mat>src_channels;
    std::vector<cv::Mat>dst_channels;
    split(small_size, src_channels);
    split(dst_part, dst_channels);
    //	CV_Assert(src_channels.size() == 4 && dst_channels.size() == 3);

    if (scale < 1)
    {
        src_channels[3] *= scale;
        scale = 1;
    }
    for (int i = 0; i < 3; i++)
    {
        dst_channels[i] = dst_channels[i].mul(255.0 / scale - src_channels[3], scale / 255.0);
        dst_channels[i] += src_channels[i].mul(src_channels[3], scale / 255.0);
    }
    merge(dst_channels, dst_part);
    return true;
}

void QtWidgetsApplication1::OpenVideoCapture() {

    CascadeClassifier detector1, detector2;
    detector2.load("haarcascade_eye_tree_eyeglasses.xml");
    detector1.load("haarcascade_frontalface_default.xml");
    Mat img2 = imread("ClownGlasses.png", -1);
    VideoCapture cap(0);
    Mat frame,frame2;
    imshow("help", imread("help.jpg"));
    while (true) {
        bool ret = cap.read(frame);
        if (!ret) {
            QMessageBox::warning(NULL, "error", "can't find camera");
            break;
        }
        frame.copyTo(frame2);
        rectangle(frame, Rect(192,70,280,350), Scalar(0, 0, 255), 2, 8);
        if (waitKey(1) == 'q') { //27 = esc
            QMessageBox::warning(NULL, "close", "close");
            //ui.image1->clear();
            destroyWindow("help");
            break;
        }
        imshow("help", frame);
    }
    QImage Qcamera = cvMat2QImage(frame2);
    ui.image1->setPixmap(QPixmap::fromImage(Qcamera));
    ui.image1->resize(ui.image1->pixmap()->size());
    ui.Width->setText("寬：" + QVariant(ui.image1->width()).toString());
    ui.Height->setText("高：" + QVariant(ui.image1->height()).toString());
}

void QtWidgetsApplication1::decoration1() {
    Mat frame = QImage2cvMat(ui.image1->pixmap()->toImage());
    Mat img1 = imread("ClownGlasses.png", -1);
    CascadeClassifier detector;
    //眼部偵測
    detector.load("haarcascade_eye_tree_eyeglasses.xml");
    if (!detector.load("haarcascade_eye_tree_eyeglasses.xml")) {
        QMessageBox::warning(NULL, "MessageBox", "error");
    }
    int x, y;
    std::vector<Rect> eyes;
    detector.detectMultiScale(frame, eyes, 1.1, 3, 0, Size(30, 30));
    //for (size_t t = 0; t < eyes.size(); t++) {
    //    rectangle(frame, eyes[t], Scalar(0, 0, 255), 2, 8);
    //    std::cout << eyes[t] << std::endl;//兩個框的左上座標
    //}
    if (eyes[0].x > eyes[1].x) {
        x = eyes[1].x;
        y = eyes[1].y;
    }
    else {
        x = eyes[0].x;
        y = eyes[0].y;
    }
    addphoto(frame, img1, 1, 0.8, 0, Point(x-39, y-40));
    imshow("test",frame);
}

void QtWidgetsApplication1::decoration2() {
    Mat frame = QImage2cvMat(ui.image1->pixmap()->toImage());
    Mat img1 = imread("dog1.png", -1);
    Mat img2 = imread("dog2.png", -1);
    CascadeClassifier detector;
    detector.load("haarcascade_frontalface_default.xml");
    if (!detector.load("haarcascade_frontalface_default.xml")) {
        QMessageBox::warning(NULL, "MessageBox", "error");
    }
    std::vector<Rect> faces;
    detector.detectMultiScale(frame, faces, 1.1, 3, 0, Size(30, 30));
    addphoto(frame, img1, 1, 0.8, 0, Point(faces[0].x-10, faces[0].y-100));
    addphoto(frame, img2, 1, 0.8, 0, Point(faces[0].x+35, faces[0].y+80));
    imshow("test", frame);
}

void QtWidgetsApplication1::decoration3() {
    Mat frame = QImage2cvMat(ui.image1->pixmap()->toImage());
    Mat img1 = imread("SantaClaus1.png", -1);
    Mat img2 = imread("SantaClaus2.png", -1);
    CascadeClassifier detector;
    detector.load("haarcascade_frontalface_default.xml");
    if (!detector.load("haarcascade_frontalface_default.xml")) {
        QMessageBox::warning(NULL, "MessageBox", "error");
    }
    std::vector<Rect> faces;
    detector.detectMultiScale(frame, faces, 1.1, 3, 0, Size(30, 30));
    addphoto(frame, img1, 1, 1, 0, Point(faces[0].x, faces[0].y - 100));
    addphoto(frame, img2, 1, 1, 0, Point(faces[0].x-30, faces[0].y + 100));
    imshow("test", frame);
}