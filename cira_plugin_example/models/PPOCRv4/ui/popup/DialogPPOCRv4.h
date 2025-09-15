#ifndef DialogPPOCRv4_H
#define DialogPPOCRv4_H

#include <QDialog>
#include <QJsonObject>
#include <QEvent>

namespace Ui {
class DialogPPOCRv4;
}

class DialogPPOCRv4 : public QDialog
{
  Q_OBJECT

public:
  explicit DialogPPOCRv4(QWidget *parent = 0);
  ~DialogPPOCRv4();

  void changeEvent(QEvent *ev) {
    if (ev->type() == QEvent::ActivationChange)
    {
        if(this->isActiveWindow())
        {

        }
        else
        {
          if(!isPin) {
            this->close();
          }
        }
    }
  }

  bool isPin = true;

  QString style_unpin = "background-color: transparent; border-image: url(:/cira_plugin_example/icon/unpin.png); background: none; border: none; background-repeat: none;";
  QString style_pin = "background-color: transparent; border-image: url(:/cira_plugin_example/icon/pin.png); background: none; border: none; background-repeat: none;";

  QJsonObject saveState();
  void restoreState(QJsonObject param_js_data);

private slots:
  void on_buttonApply_clicked();
  void on_buttonReset_clicked();
  void on_sliderDBThresh_valueChanged(int value);
  void on_spinDBThresh_valueChanged(double value);
  void on_sliderBoxThresh_valueChanged(int value);
  void on_spinBoxThresh_valueChanged(double value);
  void on_sliderUnclipRatio_valueChanged(int value);
  void on_spinUnclipRatio_valueChanged(double value);


private:
  Ui::DialogPPOCRv4 *ui;
};

#endif // DialogPPOCRv4_H
