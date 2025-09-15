#ifndef FormPPOCRv4_H
#define FormPPOCRv4_H

#include <QWidget>
#include <iostream>
#include <QMovie>
#include <QScreen>

#include "ThreadPPOCRv4.hpp"
#include <cira_lib_bernoulli/general/GlobalData.hpp>

class DialogPPOCRv4;

namespace Ui {
class FormPPOCRv4;
}

class FormPPOCRv4 : public QWidget
{
  Q_OBJECT

public:
  explicit FormPPOCRv4(QWidget *parent = 0);
  ~FormPPOCRv4();

  bool nodeStatus_enable = true;
  bool nodeStatus_ready = true;
  bool nodeStatus_complete = true;

  qint64 timestamp_base = 0;
  qint64 timestamp_input = -1;

  bool isHaveError = false;

  QString style_nodeDisable = "background-color: transparent; border-image: url(:/cira_plugin_example/icon/node_disable.png); background: none; border: none; background-repeat: none;";
  QString style_nodeEnable = "background-color: transparent; border-image: url(:/cira_plugin_example/icon/node_enable.png); background: none; border: none; background-repeat: none;";

  QMovie *mv_node_run;

  ThreadPPOCRv4 *workerThread;
  DialogPPOCRv4 *dialog;

public slots:
  void update_ui();

private slots:
  void stop_node_process();
  void on_pushButton_nodeEnable_clicked();
  void on_pushButton_prop_clicked();

private:
  Ui::FormPPOCRv4 *ui;
};

#endif // FormPPOCRv4_H
