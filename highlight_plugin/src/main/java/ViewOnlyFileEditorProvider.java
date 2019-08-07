import com.intellij.codeHighlighting.BackgroundEditorHighlighter;
import com.intellij.openapi.fileEditor.*;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.util.Key;
import com.intellij.openapi.vfs.VirtualFile;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import javax.swing.*;
import java.beans.PropertyChangeListener;

public class ViewOnlyFileEditorProvider implements FileEditorProvider {

    public static final String EDITOR_TYPE_ID = "ReadOnlyEditor";

    @Override
    public boolean accept(@NotNull Project project, @NotNull VirtualFile file) {
        return file.getName().contains(".py");
    }

    @NotNull
    @Override
    public FileEditor createEditor(@NotNull Project project, @NotNull VirtualFile file) {
        return new FileEditor() {
            @NotNull
            @Override
            public JComponent getComponent() {
                return null;
            }

            @Nullable
            @Override
            public JComponent getPreferredFocusedComponent() {
                return null;
            }

            @NotNull
            @Override
            public String getName() {
                return null;
            }

            @Override
            public void setState(@NotNull FileEditorState state) {

            }

            @Override
            public boolean isModified() {
                return false;
            }

            @Override
            public boolean isValid() {
                return false;
            }

            @Override
            public void selectNotify() {

            }

            @Override
            public void deselectNotify() {

            }

            @Override
            public void addPropertyChangeListener(@NotNull PropertyChangeListener listener) {

            }

            @Override
            public void removePropertyChangeListener(@NotNull PropertyChangeListener listener) {

            }

            @Nullable
            @Override
            public BackgroundEditorHighlighter getBackgroundHighlighter() {
                return null;
            }

            @Nullable
            @Override
            public FileEditorLocation getCurrentLocation() {
                return null;
            }

            @Override
            public void dispose() {

            }

            @Nullable
            @Override
            public <T> T getUserData(@NotNull Key<T> key) {
                return null;
            }

            @Override
            public <T> void putUserData(@NotNull Key<T> key, @Nullable T value) {

            }
        };
    }

    @NotNull
    @Override
    public String getEditorTypeId() {
        return EDITOR_TYPE_ID;
    }

    @NotNull
    @Override
    public FileEditorPolicy getPolicy() {
        return FileEditorPolicy.NONE;
    }
}
