#include "FormPPOCRv4.h"
#include "ui_FormPPOCRv4.h"

void FormPPOCRv4::update_ui() {

  if(!nodeStatus_complete) {

    mv_node_run->setScaledSize(QSize(ui->label_nodeStatus->width(), ui->label_nodeStatus->height()));
    ui->label_nodeStatus->setMovie(mv_node_run);
    mv_node_run->start();

  } else {
    mv_node_run->stop();
    if(nodeStatus_ready && !isHaveError) {
      ui->label_nodeStatus->setPixmap(QPixmap(QString::fromUtf8(":/cira_plugin_example/icon/yellow_led.png")));
    } else {
      ui->label_nodeStatus->setPixmap(QPixmap(QString::fromUtf8(":/cira_plugin_example/icon/red_led.png")));
    }
  }

  if(nodeStatus_enable) {
    ui->pushButton_nodeEnable->setStyleSheet(style_nodeEnable);
  } else {
    ui->pushButton_nodeEnable->setStyleSheet(style_nodeDisable);
  }

}

void FormPPOCRv4::stop_node_process() {
  if(!GlobalData::isAllSceneCanRun && !nodeStatus_complete) {
    if(workerThread->isRunning()) {workerThread->terminate();}
    isHaveError = true;
    update_ui();
  }
}
