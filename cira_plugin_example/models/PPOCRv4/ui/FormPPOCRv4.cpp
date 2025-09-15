#include "FormPPOCRv4.h"
#include "ui_FormPPOCRv4.h"         // ADD THIS - Generated UI header
#include "popup/DialogPPOCRv4.h"
#include "ThreadPPOCRv4.hpp"
#include <QApplication>             // ADD THIS
#include <QDesktopWidget>           // ADD THIS
#include <QCursor>                  // ADD THIS

FormPPOCRv4::FormPPOCRv4(QWidget *parent) :
  QWidget(parent),
  ui(new Ui::FormPPOCRv4)
{
  ui->setupUi(this);

  timestamp_base = GlobalData::timer.nsecsElapsed();
  nodeStatus_ready = true;

  workerThread = new ThreadPPOCRv4();
  dialog = new DialogPPOCRv4();

  mv_node_run = new QMovie(":/cira_plugin_example/icon/run_led.gif");
  update_ui();
  connect(GlobalData::GlobalDataObject, SIGNAL(stopAllScene()), this, SLOT(stop_node_process()));

  // Connect dialog accepted to save parameters to worker
  connect(dialog, &DialogPPOCRv4::accepted, this, [this]() {
    workerThread->param_js_data = dialog->saveState();
  });
}

FormPPOCRv4::~FormPPOCRv4()
{
  delete ui;
}

void FormPPOCRv4::on_pushButton_nodeEnable_clicked()
{
  if(nodeStatus_enable) {
    ui->pushButton_nodeEnable->setStyleSheet(style_nodeDisable);
    nodeStatus_enable = false;
  } else {
    ui->pushButton_nodeEnable->setStyleSheet(style_nodeEnable);
    timestamp_base = GlobalData::timer.nsecsElapsed();
    nodeStatus_enable = true;
  }
}

void FormPPOCRv4::on_pushButton_prop_clicked()
{
#ifdef WIN32
   dialog->setWindowModality(Qt::NonModal);
   dialog->setParent(GlobalData::parent);
   if(ui->label->pixmap())
      dialog->setWindowIcon(QIcon(*ui->label->pixmap()));
   dialog->setWindowFlags(Qt::Window);
#else
  dialog->setWindowModality(Qt::WindowModal);
  dialog->setWindowFlags(Qt::WindowCloseButtonHint);
#endif

  QPoint p = QCursor::pos();

  // Using QDesktopWidget for Qt 5.x
  QDesktopWidget *desktop = QApplication::desktop();
  QRect screenGeometry = desktop->screenGeometry(p);

  int w = screenGeometry.x() + screenGeometry.width();
  int h = screenGeometry.y() + screenGeometry.height();
  int x = p.x();
  int y = p.y();

  if(x + dialog->width() > w) {
    x -= x + dialog->width() - w;
  }
  if(y + dialog->height() > h) {
    y -= y + dialog->height() - h;
  }

  dialog->move(x, y);
  dialog->setVisible(true);
}
