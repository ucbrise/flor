import com.intellij.openapi.actionSystem.AnAction;
import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.actionSystem.CommonDataKeys;
import com.intellij.openapi.actionSystem.PlatformDataKeys;
import com.intellij.openapi.editor.Document;
import com.intellij.openapi.editor.EditorFactory;
import com.intellij.openapi.editor.RangeMarker;
import com.intellij.openapi.editor.ReadOnlyFragmentModificationException;
import com.intellij.openapi.editor.actionSystem.EditorActionManager;
import com.intellij.openapi.editor.actionSystem.ReadonlyFragmentModificationHandler;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.vfs.VirtualFile;
import com.intellij.psi.PsiDocumentManager;
import com.intellij.psi.PsiFile;
import org.jetbrains.annotations.NotNull;

public class ToggleReadOnlyView extends AnAction {
//    public ToggleReadOnlyView() {
//        super("Toggle Annotator View");
//    }

    @Override
    public void actionPerformed(@NotNull AnActionEvent e) {
        Project project = e.getProject();
        PsiFile file = e.getData(CommonDataKeys.PSI_FILE);
        VirtualFile virtualFile = e.getData(PlatformDataKeys.VIRTUAL_FILE);
        final EditorFactory editorFactory = EditorFactory.getInstance();


        /*
        // Create read-only editor view dialog
        Language language = JavaLanguage.INSTANCE;
        FileType fileType = language != null ? language.getAssociatedFileType() : null;
        Document myDocument = editorFactory.createDocument(file.getText());
        Editor editor = editorFactory.createViewer(myDocument, project);

        EditorTextField myTextViewer = new EditorTextField(EditorFactory.getInstance().createDocument(file.getText()), project, fileType, true, false);
        JBScrollPane scrollPane = new JBScrollPane(myTextViewer);

        try {
            ((EditorEx) editor).setHighlighter(EditorHighlighterFactory.getInstance().createEditorHighlighter(project, fileType));

            JComponent component = editor.getComponent();
            // JComponent component = myTextViewer.getComponent();
            component.setPreferredSize(new Dimension(640, 480));

            DialogBuilder dialog = new DialogBuilder(project);

            dialog.title("Highlight Viewer").dimensionKey("GuiDesigner.FormSource.Dialog");
            dialog.centerPanel(component).setPreferredFocusComponent(editor.getContentComponent());
            dialog.addOkAction();
            dialog.show();
        } finally {
            editorFactory.releaseEditor(editor);
        }

         */


        /*
        //  Make editor read-only

        PsiDocumentManager docManager = null;
        if (project != null) {
            docManager = PsiDocumentManager.getInstance(project);
        }
        Document document = null;
        if (docManager != null) {
            if (file != null) {
                document = docManager.getDocument(file);
            }
        }
        if (document != null) {
            RangeMarker guardedBlock = document.createGuardedBlock(0, document.getTextLength());
            guardedBlock.setGreedyToLeft(true);
            guardedBlock.setGreedyToRight(true);
            EditorActionManager.getInstance().setReadonlyFragmentModificationHandler(document, e1 -> {
            });
        }


         */
        toggleReadOnlyFor(project, file);

        /*
        // Toggle between read-only mode
        if (virtualFile != null) {
            WriteCommandAction.runWriteCommandAction(project, () -> {
                try {
                    virtualFile.setWritable(!virtualFile.isWritable());
                } catch (IOException ex) {
                    ex.printStackTrace();
                }
            });
        }
         */
    }

    private static void toggleReadOnlyFor(Project project, PsiFile file) {
        PsiDocumentManager docManager = null;
        if (project != null) {
            docManager = PsiDocumentManager.getInstance(project);
        }
        Document document = null;
        if (docManager != null) {
            if (file != null) {
                document = docManager.getDocument(file);
                toggleReadOnlyFor(document);
            }
        }
    }

    static void toggleReadOnlyFor(Document document) {
        if (document != null) {
            RangeMarker guardedBlock = document.createGuardedBlock(0, document.getTextLength());
            guardedBlock.setGreedyToLeft(true);
            guardedBlock.setGreedyToRight(true);
            EditorActionManager.getInstance().setReadonlyFragmentModificationHandler(document, e1 -> {
            });
        }
    }
}