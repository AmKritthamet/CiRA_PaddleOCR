#include "DialogPPOCRv4.h"
#include "ui_DialogPPOCRv4.h"

#include <QJsonObject>
#include <QJsonValue>

DialogPPOCRv4::DialogPPOCRv4(QWidget *parent) :
  QDialog(parent),
  ui(new Ui::DialogPPOCRv4)
{
  ui->setupUi(this);

  // Connect slider and spinbox pairs for synchronization
  connect(ui->sliderDBThresh, SIGNAL(valueChanged(int)), this, SLOT(on_sliderDBThresh_valueChanged(int)));
  connect(ui->spinDBThresh, SIGNAL(valueChanged(double)), this, SLOT(on_spinDBThresh_valueChanged(double)));

  connect(ui->sliderBoxThresh, SIGNAL(valueChanged(int)), this, SLOT(on_sliderBoxThresh_valueChanged(int)));
  connect(ui->spinBoxThresh, SIGNAL(valueChanged(double)), this, SLOT(on_spinBoxThresh_valueChanged(double)));

  connect(ui->sliderUnclipRatio, SIGNAL(valueChanged(int)), this, SLOT(on_sliderUnclipRatio_valueChanged(int)));
  connect(ui->spinUnclipRatio, SIGNAL(valueChanged(double)), this, SLOT(on_spinUnclipRatio_valueChanged(double)));

  // Connect Apply and Reset buttons
  connect(ui->buttonApply, SIGNAL(clicked()), this, SLOT(on_buttonApply_clicked()));
  connect(ui->buttonReset, SIGNAL(clicked()), this, SLOT(on_buttonReset_clicked()));
}

DialogPPOCRv4::~DialogPPOCRv4()
{
  delete ui;
}

void DialogPPOCRv4::on_buttonApply_clicked()
{
  accept();
}

void DialogPPOCRv4::on_buttonReset_clicked()
{
  // Reset to defaults
  ui->sliderDBThresh->setValue(30);         // Corresponds to 0.30
  ui->spinDBThresh->setValue(0.30);

  ui->sliderBoxThresh->setValue(60);        // Corresponds to 0.60
  ui->spinBoxThresh->setValue(0.60);

  ui->sliderUnclipRatio->setValue(150);     // Corresponds to 1.50
  ui->spinUnclipRatio->setValue(1.50);

  ui->checkUseDilation->setChecked(false);
  ui->comboScoreMode->setCurrentIndex(0);   // Fast

  ui->checkAutoScale->setChecked(true);
  ui->spinTextScale->setValue(1.0);

  ui->checkDrawBoxes->setChecked(true);
  ui->checkShowConfidence->setChecked(true);
  ui->checkShowTextResult->setChecked(true);
  ui->checkRemoveSpace->setChecked(false);
  ui->checkRemovePunctuation->setChecked(false);
  ui->checkRecConfFilter->setChecked(false);
  ui->spinRecConfThresh->setValue(0.80);
}

void DialogPPOCRv4::on_sliderDBThresh_valueChanged(int value)
{
  double db_value = value / 100.0;
  if (qAbs(ui->spinDBThresh->value() - db_value) > 0.001)
      ui->spinDBThresh->setValue(db_value);
}

void DialogPPOCRv4::on_spinDBThresh_valueChanged(double value)
{
  int slider_value = static_cast<int>(value * 100);
  if (ui->sliderDBThresh->value() != slider_value)
      ui->sliderDBThresh->setValue(slider_value);
}

void DialogPPOCRv4::on_sliderBoxThresh_valueChanged(int value)
{
  double db_value = value / 100.0;
  if (qAbs(ui->spinBoxThresh->value() - db_value) > 0.001)
      ui->spinBoxThresh->setValue(db_value);
}

void DialogPPOCRv4::on_spinBoxThresh_valueChanged(double value)
{
  int slider_value = static_cast<int>(value * 100);
  if (ui->sliderBoxThresh->value() != slider_value)
      ui->sliderBoxThresh->setValue(slider_value);
}

void DialogPPOCRv4::on_sliderUnclipRatio_valueChanged(int value)
{
  double db_value = value / 100.0;
  if (qAbs(ui->spinUnclipRatio->value() - db_value) > 0.001)
      ui->spinUnclipRatio->setValue(db_value);
}

void DialogPPOCRv4::on_spinUnclipRatio_valueChanged(double value)
{
  int slider_value = static_cast<int>(value * 100);
  if (ui->sliderUnclipRatio->value() != slider_value)
      ui->sliderUnclipRatio->setValue(slider_value);
}

QJsonObject DialogPPOCRv4::saveState() {
  QJsonObject param_js_data;
  param_js_data["db_thresh"]        = ui->spinDBThresh->value();
  param_js_data["box_thresh"]       = ui->spinBoxThresh->value();
  param_js_data["unclip_ratio"]     = ui->spinUnclipRatio->value();
  param_js_data["use_dilation"]     = ui->checkUseDilation->isChecked();
  param_js_data["score_mode"]       = ui->comboScoreMode->currentIndex(); // 0: Fast, 1: Slow
  param_js_data["auto_scale_text"]  = ui->checkAutoScale->isChecked();
  param_js_data["text_scale_factor"] = ui->spinTextScale->value();

  param_js_data["draw_boxes"]       = ui->checkDrawBoxes->isChecked();
  param_js_data["show_confidence"]  = ui->checkShowConfidence->isChecked();
  param_js_data["show_text"]        = ui->checkShowTextResult->isChecked();
  param_js_data["remove_space"]     = ui->checkRemoveSpace->isChecked();
  param_js_data["remove_punct"]     = ui->checkRemovePunctuation->isChecked();
  param_js_data["use_rec_conf_filter"] = ui->checkRecConfFilter->isChecked();
  param_js_data["rec_conf_thresh"] = ui->spinRecConfThresh->value();


  return param_js_data;
}

void DialogPPOCRv4::restoreState(QJsonObject param_js_data) {
  if(param_js_data.contains("db_thresh"))
      ui->spinDBThresh->setValue(param_js_data["db_thresh"].toDouble());
  if(param_js_data.contains("box_thresh"))
      ui->spinBoxThresh->setValue(param_js_data["box_thresh"].toDouble());
  if(param_js_data.contains("unclip_ratio"))
      ui->spinUnclipRatio->setValue(param_js_data["unclip_ratio"].toDouble());
  if(param_js_data.contains("use_dilation"))
      ui->checkUseDilation->setChecked(param_js_data["use_dilation"].toBool());
  if(param_js_data.contains("score_mode"))
      ui->comboScoreMode->setCurrentIndex(param_js_data["score_mode"].toInt());
  if(param_js_data.contains("auto_scale_text"))
        ui->checkAutoScale->setChecked(param_js_data["auto_scale_text"].toBool());
  if(param_js_data.contains("text_scale_factor"))
        ui->spinTextScale->setValue(param_js_data["text_scale_factor"].toDouble());

  if(param_js_data.contains("draw_boxes"))
      ui->checkDrawBoxes->setChecked(param_js_data["draw_boxes"].toBool());
  if(param_js_data.contains("show_confidence"))
      ui->checkShowConfidence->setChecked(param_js_data["show_confidence"].toBool());
  if(param_js_data.contains("show_text"))
      ui->checkShowTextResult->setChecked(param_js_data["show_text"].toBool());
  if(param_js_data.contains("remove_space"))
      ui->checkRemoveSpace->setChecked(param_js_data["remove_space"].toBool());
  if(param_js_data.contains("remove_punct"))
      ui->checkRemovePunctuation->setChecked(param_js_data["remove_punct"].toBool());
  if(param_js_data.contains("use_rec_conf_filter"))
      ui->checkRecConfFilter->setChecked(param_js_data["use_rec_conf_filter"].toBool());
  if(param_js_data.contains("rec_conf_thresh"))
      ui->spinRecConfThresh->setValue(param_js_data["rec_conf_thresh"].toDouble());
}
