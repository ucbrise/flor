import com.intellij.codeInsight.navigation.actions.GotoDeclarationOnlyAction;
import com.intellij.execution.process.ProcessHandler;
import com.intellij.openapi.actionSystem.AnAction;
import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.actionSystem.CommonDataKeys;
import com.intellij.openapi.application.ApplicationManager;
import com.intellij.openapi.editor.CaretModel;
import com.intellij.openapi.editor.Editor;
import com.intellij.openapi.editor.ScrollType;
import com.intellij.openapi.fileEditor.FileEditor;
import com.intellij.openapi.fileEditor.FileEditorManager;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.vfs.LocalFileSystem;
import com.intellij.openapi.vfs.VirtualFile;
import com.intellij.psi.PsiElement;
import com.intellij.psi.PsiFile;
import com.intellij.psi.PsiManager;
import com.intellij.codeInsight.navigation.actions.GotoDeclarationAction;
//import com.jetbrains.python.psi.impl.PyGotoDeclarationHandler;
import org.jetbrains.annotations.NotNull;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import java.io.*;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class FindAndCloneSource extends AnAction {
    @Override
    public void actionPerformed(@NotNull AnActionEvent e) {
        Project project = e.getProject();
        Editor editor = e.getData(CommonDataKeys.EDITOR);
        PsiFile file = e.getData(CommonDataKeys.PSI_FILE);
        VirtualFile vFile = file.getVirtualFile();
        String projectBasePath = null;
        if (project != null) {
            projectBasePath = project.getBasePath();
        }
        String currFolderName = vFile.getParent().getName();


//        Open the raw script that implements the definition of the selection
        CaretModel caretModel = editor.getCaretModel();
        String selectedText = caretModel.getCurrentCaret().getSelectedText();
        int caretOffset = caretModel.getOffset();
        PsiElement element = file.findElementAt(caretOffset);
        // PyGotoDeclarationHandler handler = new PyGotoDeclarationHandler();
        PsiElement targetElement = GotoDeclarationAction.findTargetElement(project, editor, caretOffset);
        // PsiElement elementReference = handler.getGotoDeclarationTarget(element, editor);
        // PsiReference elementReference = element.getReference();
        // PsiMethod containingMethod = PsiTreeUtil.getParentOfType(element, PsiMethod.class);

        // Open Source File if the selection has one
        if (targetElement != null) {
            VirtualFile srcVFile = targetElement.getContainingFile().getVirtualFile();
            String srcFilePath = srcVFile.getPath();
            String srcFolderPath = srcVFile.getParent().getPath();

            // TODO: Fit more general files from other python libs
            // ---------------
            // START: Fit more general files from other python libs
            // ---------------
            String[] srcFilePathSplit = srcFilePath.split("/");

            String srcFileNameRoot = srcVFile.getName().split(".py")[0];

            List<String> srcFileNameToJoin = new ArrayList<>();
            // TODO: If "site-packages" is in split join everything after that to form the root name
            if (Arrays.asList(srcFilePathSplit).contains("site-packages")) {
                int index = Arrays.asList(srcFilePathSplit).indexOf("site-packages");
                for (int i = index + 1; i < srcFilePathSplit.length - 1; i++) {
                    srcFileNameToJoin.add(srcFilePathSplit[i]);
                }

                srcFileNameRoot = String.join("_", srcFileNameToJoin) + "_" + srcFileNameRoot;
            }

            // END: If "site-packages" is in split join everything after that to form the root name

            // String dstFilePath = projectBasePath + "/" + srcFileNameRoot + "_h.py";
            Date date = new Date();
            long tsMilliseconds = date.getTime();
            String timeStamp = new SimpleDateFormat("yyyyMMdd-HHmmss").format(new Date());

            String dstFolderPath = projectBasePath + "/" + "highlighted." + timeStamp + ".d";

            // DONE: get dstFolderPath from .flor_highlight/config.xml
            File dstFlorPluginFolder = new File(projectBasePath + "/.flor_highlight");
            String configFilePath = dstFlorPluginFolder + "/config.xml";
            File configFile = new File(configFilePath);

            if (!configFile.exists()) {
                new StartNewHLSession().actionPerformed(e);
                System.out.println("No session found! Auto starting a new session...");
            }

            DocumentBuilderFactory documentFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder documentBuilder = null;
            try {
                documentBuilder = documentFactory.newDocumentBuilder();
            } catch (ParserConfigurationException ex) {
                ex.printStackTrace();
            }

            int numSessions = 0;

            try {
                Document document = documentBuilder.parse(configFile);
                Element root = document.getDocumentElement();
                if (root != null) {
                    numSessions = Integer.parseInt(root.getAttribute("numSessions"));
                }
                Element session = null;
                if (root != null) {
                    NodeList sessionList = document.getElementsByTagName("session");
                    for (int i = 0; i < sessionList.getLength(); i++) {
                        Element s = (Element) sessionList.item(i);
                        s.setIdAttribute("id", true);
                    }
                    session = document.getElementById(String.valueOf(numSessions));
                }

                if (session != null) {
                    dstFolderPath = session.getElementsByTagName("path").item(0).getFirstChild().getTextContent();
                }
            } catch (SAXException | IOException ex) {
                ex.printStackTrace();
            }
            // ---------------
            // END: get dstFolderPath from .flor_highlight/config.xml
            // ---------------

            File dstDir = new File(dstFolderPath);
            dstDir.mkdir();
            String dstFilePath = dstFolderPath + "/" + srcFileNameRoot + "_h.py";

            String program = "from argparse import Namespace; from flor.commands.cp import exec_cp; exec_cp(Namespace(src='" + srcFilePath + "', dst='" + dstFilePath + "'))";
            Process p = null;
            try {
                File tempFile = File.createTempFile("flor_plugin_temp", ".py");
                BufferedWriter out = new BufferedWriter(new FileWriter(tempFile));
                out.write(program);
                out.close();
                p = Runtime.getRuntime().exec("python " + tempFile.getAbsolutePath());

                BufferedReader in = new BufferedReader(new InputStreamReader(p.getInputStream()));
                String ret = in.readLine();
                System.out.println("RetVal: " + ret);
            } catch (IOException ex) {
                ex.printStackTrace();
            }

            File dstFile = new File(dstFilePath);

            ProcessHandler processHandler = null;

            // Attempt to generate another time if method 1 does not work
            if (!dstFile.exists()) {

                File tempFile = null;
                try {
                    tempFile = File.createTempFile("flor_plugin_temp", ".py");
                    BufferedWriter out = new BufferedWriter(new FileWriter(tempFile));
                    out.write(program);
                    out.close();
                } catch (IOException ex) {
                    ex.printStackTrace();
                }

                // https://stackoverflow.com/questions/36853427/intellij-plugin-run-console-command
                ArrayList<String> command = new ArrayList<>();
                command.add("/bin/bash");
                command.add("-c");
                if (tempFile != null) {
                    command.add("python " + tempFile.getAbsolutePath());
                }

                if (project != null) {
                    processHandler = RunCommandLine.getProcessHandler(command, project.getBasePath());
                }
            }

            // Must wait for the process to finish
            if (processHandler != null) {
                processHandler.waitFor();
            }

            if (!dstFile.exists()) {
                return;
            }

            VirtualFile dstVFile = LocalFileSystem.getInstance().refreshAndFindFileByIoFile(dstFile);
            // PsiFile dstPFile = PsiManager.getInstance(project).findFile(dstVFile);

            // Open Source File in new editor tab
            // FileEditorManager.getInstance(project).openFile(srcVFile, true);
            if (dstVFile != null) {
                assert project != null : "project is null";
                FileEditorManager.getInstance(project).openFile(dstVFile, true);
                Editor selectedTextEditor = FileEditorManager.getInstance(project).getSelectedTextEditor();
                if (selectedTextEditor != null) {
                    selectedTextEditor.getCaretModel().moveToOffset((int) (targetElement.getTextOffset() + dstVFile.getLength() - srcVFile.getLength()));
                    selectedTextEditor.getScrollingModel().scrollToCaret(ScrollType.CENTER);
                    ToggleReadOnlyView.toggleReadOnlyFor(selectedTextEditor.getDocument());
                }
            }
        }

    }
}
