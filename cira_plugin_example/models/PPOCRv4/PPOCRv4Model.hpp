#ifndef PPOCRv4Model_HPP
#define PPOCRv4Model_HPP

#pragma once

#include <QtCore/QObject>
#include <QtCore/QJsonObject>
#include <QtCore/QJsonValue>
#include <QWidget>
#include <QTimer>
#include <QDateTime>
#include <QShowEvent>

#include <cira_lib_bernoulli/general/GlobalData.hpp>

#include <nodes/NodeDataModel>

#include <iostream>

// Forward declarations
class FormPPOCRv4;
class FlowData;

using QtNodes::PortType;
using QtNodes::PortIndex;
using QtNodes::NodeData;
using QtNodes::NodeDataType;
using QtNodes::NodeDataModel;
using QtNodes::NodeValidationState;

class PPOCRv4Model : public NodeDataModel
{
  Q_OBJECT

public:
  PPOCRv4Model();

  virtual
  ~PPOCRv4Model() {}

  int portInStatus[1] = {PORTSTATUS::DISCONNECTED};

public:

  QString
  caption() const override
  { return QStringLiteral("PPOCRv4"); }

  bool
  captionVisible() const override
  { return true; }

  bool
  portCaptionVisible(PortType, PortIndex) const override
  { return true; }

  bool
  resizable() const override { return false; }

  QString
  name() const override
  { return QStringLiteral("PPOCRv4"); }

public:

  QJsonObject
  save() const override;

  void
  restore(QJsonObject const &p) override;

public:

  unsigned int
  nPorts(PortType portType) const override;

  NodeDataType
  dataType(PortType portType, PortIndex portIndex) const override;

  std::shared_ptr<NodeData>
  outData(PortIndex port) override;

  void
  setInData(std::shared_ptr<NodeData> data, PortIndex portIndex) override;

  // Declare only, implement in .cpp file
  QWidget *
  embeddedWidget() override;

private:

  bool isBusy = false;
  std::shared_ptr<FlowData> _flowDataOut;
  FormPPOCRv4 *form;

  QTimer *timerLoadFromButton;

private slots:
  void runProcess(std::shared_ptr<FlowData> _flowDataIn);
  void loopTimerLoadFromButton();

};

#endif // PPOCRv4Model_HPP
