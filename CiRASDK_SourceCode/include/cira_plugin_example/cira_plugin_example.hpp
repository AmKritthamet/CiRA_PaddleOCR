#ifndef cira_plugin_example_HPP
#define cira_plugin_example_HPP

#include <cira_lib_bernoulli/plugin/ciraloader_base.h>

class cira_plugin_example : public CiRALoader
{
  Q_OBJECT
  Q_PLUGIN_METADATA(IID "teerawat.CiRACORE.CiRALoader")
  Q_INTERFACES(CiRALoader)

public:
  cira_plugin_example();

  virtual std::shared_ptr<DataModelRegistry> registerDataModels(std::shared_ptr<DataModelRegistry> ret) override;

  virtual void setTreeWidget() override;

  virtual const QString name() const override
  {
    return info()["name"].toString();
  }

  virtual const QJsonObject info() const override
  {
    QJsonObject jso;
    jso["name"] = "cira_plugin_example";
    jso["version"] = "0.0.0";
    jso["maintainer"] = "anonymous";
    jso["email"] = "";

    return jso;
  }

};

#endif // cira_plugin_example_HPP
