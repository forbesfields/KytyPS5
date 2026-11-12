#include "configuration.h"
#include "configurationEditDialog.h"

#include <QApplication>
#include <QCheckBox>
#include <QPushButton>
#include <QPixmap>
#include <QScrollArea>
#include <QTimer>
#include <QJsonObject>
#include <QSettings>
#include <QTemporaryDir>

#include <cstdio>
#include <cstdlib>

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "LauncherOutput1080pTests: %s\n", message);
		std::exit(1);
	}
}

} // namespace

int main(int argc, char* argv[]) {
	qputenv("QT_QPA_PLATFORM", "offscreen");
	QApplication app(argc, argv);
	QTimer::singleShot(15000, [] { Check(false, "dialog test timed out"); });
	QTemporaryDir directory;
	Check(directory.isValid(), "temporary settings directory");
	const auto path = directory.filePath(QStringLiteral("Kyty.ini"));
	{
		QSettings settings(path, QSettings::IniFormat);
		Configuration legacy;
		legacy.output_1080p_enabled = true;
		legacy.ReadSettings(&settings);
		Check(!legacy.output_1080p_enabled, "a missing INI key must default off");
	}

	for (const bool enabled: {true, false}) {
		Configuration source;
		source.output_1080p_enabled = enabled;
		{
			QSettings settings(path, QSettings::IniFormat);
			source.WriteSettings(&settings);
			settings.sync();
			Check(settings.status() == QSettings::NoError, "write the INI file");
		}
		{
			QSettings settings(path, QSettings::IniFormat);
			Configuration restored;
			restored.ReadSettings(&settings);
			Check(restored.output_1080p_enabled == enabled, "INI bool round trip");
			Check(settings.value("output_1080p_enabled").toBool() == enabled,
			      "save the documented INI key");
		}
		Configuration copied;
		copied.CopyEmulatorSettingsFrom(source);
		Check(copied.output_1080p_enabled == enabled, "global and per-game settings copy");
		Configuration imported;
		QString error;
		const QJsonObject json {{QStringLiteral("output_1080p_enabled"),
		                         enabled ? QStringLiteral("true") : QStringLiteral("false")}};
		Check(imported.SetGameSettings(json, error), "accept the game settings key");
		Check(imported.output_1080p_enabled == enabled, "JSON bool round trip");
	}
	Configuration edited;
	for (const bool enabled: {true, false}) {
		ConfigurationEditDialog dialog(edited);
		auto* checkbox = dialog.findChild<QCheckBox*>("checkBox_output_1080p");
		auto* save = dialog.findChild<QPushButton*>("ok_button");
		Check(checkbox != nullptr && save != nullptr, "display checkbox and Save button exist");
		Check(checkbox->isChecked() == edited.output_1080p_enabled, "restore the saved checkbox");
		dialog.show();
		app.processEvents();
		checkbox->click();
		Check(checkbox->isChecked() == enabled, "toggle the display checkbox");
		if (enabled && argc == 2) {
			for (auto* scroll: dialog.findChildren<QScrollArea*>()) {
				if (scroll->isAncestorOf(checkbox)) {
					scroll->ensureWidgetVisible(checkbox);
				}
			}
			app.processEvents();
			Check(dialog.grab().save(QString::fromLocal8Bit(argv[1])), "save the settings capture");
		}
		save->click();
		Check(dialog.result() == QDialog::Accepted, "save the settings dialog");
		Check(edited.output_1080p_enabled == enabled, "save the changed checkbox value");
	}
	std::printf("LauncherOutput1080pTests: passed (INI default, INI/JSON round trips, copy, checkbox off/on)\n");
	return 0;
}
