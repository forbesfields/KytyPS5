#include "mainDialog.h"
#include "launcherTheme.h"

#include <QApplication>
#include <QDir>
#include <QIcon>
#include <QMessageBox>
#include <QSettings>
#include <QStandardPaths>

int main(int argc, char* argv[]) {
	QApplication a(argc, argv);
	a.setApplicationDisplayName(QStringLiteral("PuppyPS5"));
#if defined(__APPLE__)
	// Keep the existing KytyPS5 data path so PuppyPS5 uses the same saves and settings.
	// Writable state stays outside the signed app, including when macOS translocates it.
	const auto runtime = QDir(QStandardPaths::writableLocation(QStandardPaths::GenericDataLocation))
	                         .filePath(QStringLiteral("KytyPS5"));
	if (!QDir().mkpath(runtime) || !QDir::setCurrent(runtime)) {
		QMessageBox::critical(nullptr, QStringLiteral("PuppyPS5"),
		                      QStringLiteral("Cannot open the application data folder:\n%1").arg(runtime));
		return 1;
	}
	QSettings::setPath(QSettings::IniFormat, QSettings::SystemScope, runtime);
#endif
	// Title bars and taskbars get the bolder small-size artwork.
	const QIcon small_art(QStringLiteral(":/icons/kyty-small.svg"));
	const QIcon large_art(QStringLiteral(":/icons/kyty.svg"));
	QIcon       icon;
	for (int size: {16, 24, 32, 48, 64}) {
		icon.addPixmap(small_art.pixmap(QSize(size, size), 1.0));
	}
	for (int size: {128, 256}) {
		icon.addPixmap(large_art.pixmap(QSize(size, size), 1.0));
	}
	QApplication::setWindowIcon(icon);
	LauncherTheme::Initialize(a);

	MainDialog w;

	w.emit Start();

	w.show();

	return QApplication::exec();
}
